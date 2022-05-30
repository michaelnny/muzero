from absl import logging
from typing import Callable, Iterable, List, Tuple, Mapping, Text, Any
from pathlib import Path
import sys
import signal
import copy
import pickle
import time
import queue
import multiprocessing
import gym

import numpy as np
import torch
import torch.nn.functional as F

from muzero.games.env import BoardGameEnv
from muzero.network import MuZeroNet
from muzero.replay import Transition, PrioritizedReplay
from muzero.trackers import make_self_play_trackers, make_learner_trackers, make_evaluator_trackers
from muzero.mcts import uct_search
from muzero.util import scalar_to_categorical_probabilities, logits_to_transformed_expected_value
from muzero.rating import compute_elo_rating


@torch.no_grad()
def run_self_play(
    rank: int,
    network: MuZeroNet,
    device: torch.device,
    env: gym.Env,
    data_queue: multiprocessing.Queue,
    train_steps_counter: multiprocessing.Value,
    mcts_temp_func: Callable[[int, int], float],
    num_simulations: int,
    n_step: int,
    unroll_step: int,
    discount: float,
    pb_c_base: float,
    pb_c_init: float,
    root_noise_alpha: float,
    min_bound: float,
    max_bound: float,
    stop_event: multiprocessing.Event,
    tag: str = None,
    use_tensorboard: bool = True,
    is_board_game: bool = False,
) -> None:
    """Run self-play for as long as needed, only stop if stop_event is set to True."""

    init_absl_logging()
    handle_exit_signal()
    logging.info(f'Start self-play actor {rank}')

    tb_log_dir = f'actor{rank}'
    if tag is not None and tag != '':
        tb_log_dir = f'{tag}_{tb_log_dir}'

    trackers = make_self_play_trackers(tb_log_dir) if use_tensorboard else []
    for tracker in trackers:
        tracker.reset()

    network = network.to(device=device)
    network.eval()
    game = 0

    while not stop_event.is_set():
        # For each new game.
        obs = env.reset()
        done = False
        episode_trajectory = []

        # Play and record transitions.
        while not done:
            # Make a copy of current player id.
            player_id = copy.deepcopy(env.current_player)

            action, pi_prob, root_value, value_error = uct_search(
                state=obs,
                network=network,
                device=device,
                discount=discount,
                pb_c_base=pb_c_base,
                pb_c_init=pb_c_init,
                temperature=mcts_temp_func(env.steps, train_steps_counter.value),
                num_simulations=num_simulations,
                root_noise_alpha=root_noise_alpha,
                actions_mask=env.actions_mask,
                current_player=env.current_player,
                opponent_player=env.opponent_player,
                is_board_game=is_board_game,
                min_bound=min_bound,
                max_bound=max_bound,
            )

            next_obs, reward, done, _ = env.step(action)

            for tracker in trackers:
                tracker.step(reward, done)

            episode_trajectory.append((obs, action, reward, pi_prob, root_value, value_error, player_id))

            obs = next_obs

            # Send samples to learner every 200 steps on Atari games.
            if not is_board_game and len(episode_trajectory) == 200 + unroll_step + n_step:
                # Unpack list of tuples into seperate lists.
                observations, actions, rewards, pi_probs, root_values, priorities, player_ids = map(
                    list, zip(*episode_trajectory)
                )
                # Compute n_step target value.
                target_values = compute_n_step_target(rewards, root_values, n_step, discount)

                # Make unroll sequences and send to learner.
                for transition, priority in make_unroll_sequence(
                    observations[:200],
                    actions[: 200 + unroll_step],
                    rewards[: 200 + unroll_step],
                    pi_probs[: 200 + unroll_step],
                    target_values[: 200 + unroll_step],
                    priorities[: 200 + unroll_step],
                    unroll_step,
                ):
                    data_queue.put((transition, priority))

                del episode_trajectory[:200]
                del (observations, actions, rewards, pi_probs, root_values, priorities, player_ids, target_values)

        game += 1
        if game % 100 == 0:
            logging.info(f'Self-play actor {rank} played {game} games')

        # Unpack list of tuples into seperate lists.
        observations, actions, rewards, pi_probs, root_values, priorities, player_ids = map(list, zip(*episode_trajectory))

        if is_board_game:
            # Using MC returns as target value.
            target_values = compute_mc_returns(rewards, player_ids)
        else:
            # Compute n_step target value.
            target_values = compute_n_step_target(rewards, root_values, n_step, discount)

        # Make unroll sequences and send to learner.
        for transition, priority in make_unroll_sequence(
            observations, actions, rewards, pi_probs, target_values, priorities, unroll_step
        ):
            data_queue.put((transition, priority))

        del episode_trajectory[:]
        del (observations, actions, rewards, pi_probs, root_values, priorities, player_ids, target_values)

    logging.info(f'Stop self-play actor {rank}')


def run_training(  # noqa: C901
    network: MuZeroNet,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,
    device: torch.device,
    actor_network: torch.nn.Module,
    replay: PrioritizedReplay,
    min_replay_size: int,
    batch_size: int,
    num_train_steps: int,
    train_steps_counter: multiprocessing.Value,
    clip_grad: bool,
    max_grad_norm: float,
    checkpoint_frequency: int,
    checkpoint_dir: str,
    checkpoint_files: List,
    stop_event: multiprocessing.Event,
    delay: int = 0,
    tag: str = None,
    is_board_game: bool = False,
):
    """Run the main training loop for N iterations, each iteration contains M updates.
    This controls the 'pace' of the pipeline, including when should the other parties to stop.

    Args:
        network: the neural network we want to optimize.
        optimizer: neural network optimizer.
        lr_scheduler: learning rate annealing scheduler.
        device: torch runtime device.
        actor_network: the neural network actors runing self-play, for the case AlphaZero pipeline without evaluation.
        replay: a simple uniform experience replay.
        min_replay_size: minimum replay size before start training.
        batch_size: sample batch size during training.
        num_iterations: number of traning iterations to run.
        updates_per_iteration: network updates per iteration.
        checkpoint_frequency: the frequency to create new checkpoint.
        checkpoint_dir: create new checkpoint save directory.
        checkpoint_files: a shared list contains the full path for the most recent new checkpoint files.
        csv_file: a csv file contains the training statistics.
        stop_event: a multiprocessing.Event signaling other parties to stop running pipeline.
        delay: wait time (in seconds) before start training on next batch samples, default 0.


    Raises:
        ValueError:
            if `num_iterations`, `updates_per_iteration`, `batch_size`, `min_replay_size`,
                or `checkpoint_frequency` is not positive integer.
            if `checkpoint_dir` is invalid.
    """

    if batch_size < 1:
        raise ValueError(f'Expect batch_size to be positive integer, got {batch_size}')
    if min_replay_size < batch_size:
        raise ValueError(f'Expect min_replay_size > batch_size, got {min_replay_size}, and {batch_size}')
    if checkpoint_frequency < 1:
        raise ValueError(f'Expect checkpoint_frequency to be positive integer, got {checkpoint_frequency}')
    if not isinstance(checkpoint_dir, str) or checkpoint_dir == '':
        raise ValueError(f'Expect checkpoint_dir to be valid path, got {checkpoint_dir}')

    logging.info('Start training thread')

    tb_log_dir = 'learner'
    ckpt_prefix = 'train_steps'

    if tag is not None and tag != '':
        tb_log_dir = f'{tag}_{tb_log_dir}'
        ckpt_prefix = f'{tag}_{ckpt_prefix}'

    trackers = make_learner_trackers(tb_log_dir)
    for tracker in trackers:
        tracker.reset()

    network = network.to(device=device)
    network.train()

    ckpt_dir = Path(checkpoint_dir)
    if checkpoint_dir is not None and checkpoint_dir != '' and not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def get_state_to_save():
        return {
            'network': network.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_steps': train_steps_counter.value,
        }

    while True:
        if replay.size < min_replay_size:
            continue

        # Signaling other parties to stop running pipeline.
        if train_steps_counter.value >= num_train_steps:
            break

        transitions, indices, weights = replay.sample(batch_size)
        weights = torch.from_numpy(weights).to(device=device, dtype=torch.float32)

        optimizer.zero_grad()
        loss, priorities = calc_loss(network, device, transitions, weights, is_board_game)
        loss.backward()

        if clip_grad:
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)

        optimizer.step()
        lr_scheduler.step()

        if priorities.shape != (batch_size,):
            raise RuntimeError(f'Expect priorities has shape ({batch_size}, ), got {priorities.shape}')
        replay.update_priorities(indices, priorities)

        train_steps_counter.value += 1

        for tracker in trackers:
            tracker.step(loss.detach().cpu().item(), lr_scheduler.get_last_lr()[0])

        del transitions, indices, weights

        if train_steps_counter.value > 1 and train_steps_counter.value % checkpoint_frequency == 0:
            state_to_save = get_state_to_save()
            ckpt_file = ckpt_dir / f'{ckpt_prefix}_{train_steps_counter.value}'
            create_checkpoint(state_to_save, ckpt_file)

            checkpoint_files.append(ckpt_file)

            actor_network.load_state_dict(network.state_dict())
            actor_network.eval()

            del state_to_save

        # Wait for sometime before start training on next batch.
        if delay is not None and delay > 0 and train_steps_counter.value > 1:
            time.sleep(delay)

    stop_event.set()


def run_evaluation(
    old_checkpoint_network: torch.nn.Module,
    new_checkpoint_network: torch.nn.Module,
    device: torch.device,
    env: BoardGameEnv,
    discount: float,
    pb_c_base: float,
    pb_c_init: float,
    root_noise_alpha: float,
    temperature: float,
    num_simulations: int,
    min_bound: float,
    max_bound: float,
    checkpoint_files: List,
    stop_event: multiprocessing.Event,
    initial_elo: int = -2000,
    tag: str = None,
) -> None:
    """Monitoring training progress by play a single game with new checkpoint againt last checkpoint.
    This is for board game only.

    Args:
        old_checkpoint_network: the last checkpoint network.
        new_checkpoint_network: new checkpoint network we want to evaluate.
        device: torch runtime device.
        env: a BoardGameEnv type environment.
        c_puct: a constant controls the level of exploration during MCTS search.
        temperature: the temperature exploration rate after MCTS search
            to generate play policy.
        num_simulations: number of simulations for each MCTS search.
        checkpoint_files: a shared list contains the full path for the most recent new checkpoint.
        stop_event: a multiprocessing.Event signaling to stop running pipeline.
        initial_elo: initial elo ratings for the players, default -2000.

     Raises:
        ValueError:
            if `env` is not a valid BoardGameEnv instance.
            if `temperature` is not a non-negative float type.
            if ``num_simulations` is not positive integer.
    """
    if not isinstance(env, BoardGameEnv):
        raise ValueError(f'Expect env to be a valid BoardGameEnv instance, got {env}')
    if not isinstance(temperature, float) or temperature <= 0.0:
        raise ValueError(f'Expect temperature to be a non-negative float, got {temperature}')
    if not isinstance(num_simulations, int) or num_simulations < 1:
        raise ValueError(f'Expect num_simulations to be a positive integer, got {num_simulations}')

    init_absl_logging()
    handle_exit_signal()
    logging.info('Start evaluation process')

    tb_log_dir = 'evaluator'

    if tag is not None and tag != '':
        tb_log_dir = f'{tag}_{tb_log_dir}'

    trackers = make_evaluator_trackers(tb_log_dir)
    for tracker in trackers:
        tracker.reset()

    disable_auto_grad(old_checkpoint_network)
    disable_auto_grad(new_checkpoint_network)

    old_checkpoint_network = old_checkpoint_network.to(device=device)
    new_checkpoint_network = new_checkpoint_network.to(device=device)

    # Set initial elo ratings
    black_elo = initial_elo
    white_elo = initial_elo

    while not stop_event.is_set():
        if len(checkpoint_files) == 0:
            continue

        # Remove the checkpoint file path from the shared list.
        ckpt_file = checkpoint_files.pop(0)
        loaded_state = load_checkpoint(ckpt_file, device)
        new_checkpoint_network.load_state_dict(loaded_state['network'])
        train_steps = loaded_state['train_steps']
        del loaded_state

        new_checkpoint_network.eval()
        old_checkpoint_network.eval()

        obs = env.reset()
        done = False

        while not done:
            # Black is the new checkpoint, white is last checkpoint.
            if env.current_player == env.black_player_id:
                network = new_checkpoint_network
            else:
                network = old_checkpoint_network

            action, *_ = uct_search(
                state=obs,
                network=network,
                device=device,
                discount=discount,
                pb_c_base=pb_c_base,
                pb_c_init=pb_c_init,
                num_simulations=num_simulations,
                root_noise_alpha=root_noise_alpha,
                actions_mask=env.actions_mask,
                current_player=env.current_player,
                opponent_player=env.opponent_player,
                min_bound=min_bound,
                max_bound=max_bound,
                is_board_game=True,
                best_action=True,
            )

            _, _, done, _ = env.step(action)

        if env.winner == env.black_player_id:
            black_elo, _ = compute_elo_rating(0, black_elo, white_elo)
        elif env.winner == env.white_player_id:
            black_elo, _ = compute_elo_rating(1, black_elo, white_elo)
        white_elo = black_elo

        for tracker in trackers:
            tracker.step(black_elo, env.steps, train_steps)

        old_checkpoint_network.load_state_dict(new_checkpoint_network.state_dict())


def run_data_collector(
    data_queue: multiprocessing.SimpleQueue,
    replay: PrioritizedReplay,
    save_frequency: int,
    save_dir: str,
    stop_event: multiprocessing.Event,
    tag: str = None,
) -> None:
    """Collect samples from self-play,
    this runs on the same process as the training loop,
    but with a separate thread.

    Args:
        data_queue: a multiprocessing.SimpleQueue to receive samples from self-play processes.
        replay: a simple uniform random experience replay.
        save_frequency: the frequency to save replay state.
        save_dir: where to save replay state.
        stop_event: a multiprocessing.Event signaling stop runing pipeline.

    """

    logging.info('Start data collector thread')

    samples_prefix = 'replay'

    if tag is not None and tag != '':
        samples_prefix = f'{tag}_{samples_prefix}'

    save_samples_dir = Path(save_dir)
    if save_dir is not None and save_dir != '' and not save_samples_dir.exists():
        save_samples_dir.mkdir(parents=True, exist_ok=True)

    should_save = save_samples_dir.exists() and save_frequency > 0

    while not stop_event.is_set():
        try:
            item = data_queue.get()
            transition, priority = item
            replay.add(transition, priority)
            if should_save and replay.num_added > 1 and replay.num_added % save_frequency == 0:
                save_file = save_samples_dir / f'{samples_prefix}_{replay.size}_{get_time_stamp(True)}'
                save_to_file(replay.get_state(), save_file)
                logging.info(f"Replay samples saved to '{save_file}'")
        except queue.Empty:
            pass
        except EOFError:
            pass


def calc_loss(
    network: MuZeroNet, device: torch.device, transitions: Transition, weights: torch.Tensor, is_board_game: bool
) -> torch.Tensor:
    """Given a network and batch of transitions, compute the loss for MuZero agent."""
    # [B, state_shape]
    state = torch.from_numpy(transitions.state).to(device=device, dtype=torch.float32, non_blocking=True)
    # [B, T]
    action = torch.from_numpy(transitions.action).to(device=device, dtype=torch.long, non_blocking=True)
    target_value_scalar = torch.from_numpy(transitions.value).to(device=device, dtype=torch.float32, non_blocking=True)
    target_reward_scalar = torch.from_numpy(transitions.reward).to(device=device, dtype=torch.float32, non_blocking=True)
    # [B, T, num_actions]
    target_pi_prob = torch.from_numpy(transitions.pi_prob).to(device=device, dtype=torch.float32, non_blocking=True)

    if not is_board_game:
        # Convert scalar targets into transformed support (probabilities).
        # [B, T, num_actions]
        target_value = scalar_to_categorical_probabilities(target_value_scalar, network.value_support_size)
        target_reward = scalar_to_categorical_probabilities(target_reward_scalar, network.reward_support_size)
    else:
        target_value = target_value_scalar
        target_reward = target_reward_scalar

    B, T = action.shape
    reward_loss, value_loss, policy_loss = (0, 0, 0)
    loss_scale = 1.0 / T
    # Placeholder for priorities, in case board game is using uniform replay, so the priorities does not matter.
    priorities = np.ones((B,)) * 1e-4

    # Get initial hidden state.
    hidden_state = network.represent(state)

    # Unroll K steps.
    for t in range(T):
        pred_pi_logits, pred_value = network.prediction(hidden_state)
        hidden_state, pred_reward = network.dynamics(hidden_state, action[:, t].unsqueeze(1))

        if t == 0 and not is_board_game:
            with torch.no_grad():
                # Using the delta between predicted value and target value (for the initial hidden state) as priorities.
                pred_value_scalar = logits_to_transformed_expected_value(pred_value, network.value_supports).squeeze(1)
                priorities = torch.abs(pred_value_scalar - target_value_scalar[:, 0]).cpu().numpy()

        # Scale the gradient for dynamics function by 0.5.
        hidden_state.register_hook(lambda grad: grad * 0.5)

        value_loss += loss_func(pred_value.squeeze(), target_value[:, t], is_board_game)
        if not is_board_game:
            reward_loss += loss_func(pred_reward.squeeze(), target_reward[:, t], is_board_game)
        # policy_loss += loss_func(network_output.pi_logits, target_pi_prob[:, t])
        policy_loss += F.cross_entropy(pred_pi_logits, target_pi_prob[:, t], reduction='none')

    # Board game uses uniform replay, we skip the sample weights.
    if not is_board_game:
        reward_loss = reward_loss * weights.detach()
        value_loss = value_loss * weights.detach()
        policy_loss = policy_loss * weights.detach()

    loss = torch.mean(reward_loss + value_loss + policy_loss)

    # Scale the loss by 1/unroll_step.
    loss.register_hook(lambda grad: grad * loss_scale)

    return loss, priorities


def loss_func(prediction: torch.Tensor, target: torch.Tensor, mse: bool = False) -> torch.Tensor:
    """Loss function for MuZero agent."""
    assert prediction.shape == target.shape

    if not mse:
        assert len(prediction.shape) == 2

    if mse:
        # [B]
        # return 0.5 * torch.square((prediction - target))
        return F.mse_loss(prediction, target, reduction='none')

    # [B]
    # return torch.sum(-target * F.log_softmax(prediction, dim=1), dim=1)
    return F.cross_entropy(prediction, target, reduction='none')


def compute_n_step_target(in_rewards: List[float], in_root_values: List[float], n_step: int, discount: float) -> List[float]:
    """Compute n-step target for Atari and classic openAI Gym problems.

    Args:
        rewards: a list of rewards received from the env, length T.
        root_values: a list of root node value from MCTS search, length T.
        n_step: the number of steps into the future for n-step value.
        discount: discount for future reward.

    Returns:
        a list of n-step target value, length T.

    Raises:
        ValueError:
            lists `rewards` and `root_values` do not have equal length.
    """

    if len(in_rewards) != len(in_root_values):
        raise ValueError('Arguments `rewards` and `root_values` don have the same length.')

    T = len(in_rewards)

    rewards = list(in_rewards)
    root_values = list(in_root_values)

    # Padding zeros at the end of trajectory for easy computation.
    rewards += [0] * n_step
    root_values += [0] * n_step

    target_values = []
    for t in range(T):
        bootstrap_index = t + n_step
        n_step_target = sum([r * discount**i for i, r in enumerate(rewards[t:bootstrap_index])])

        # Add MCTS root node search value for bootstrap.
        n_step_target += root_values[bootstrap_index]
        target_values.append(n_step_target)

    return target_values


def compute_mc_returns(rewards: List[float], player_ids: List[float]) -> List[float]:
    """Compute the target value using Monte Carlo returns.

    Args:
        rewards: a list of rewards received from the env, length T.
        player_ids: a list of player id for each of the transition, length T.

    Returns:
        a list of target value using MC return, length T.

    Raises:
        ValueError:
            lists `rewards` and `player_ids` do not have equal length.
    """
    if len(rewards) != len(player_ids):
        raise ValueError('Arguments `rewards` and `player_ids` don have the same length.')

    T = len(rewards)

    target_values = [0.0] * T

    final_reward = rewards[-1]
    final_player = player_ids[-1]

    if final_reward != 0.0:
        for t in range(T):
            if player_ids[t] == final_player:
                target_values[t] = final_reward
            else:
                target_values[t] = -final_reward

    return target_values


def make_unroll_sequence(
    observations: List[np.ndarray],
    actions: List[int],
    rewards: List[float],
    pi_probs: List[np.ndarray],
    values: List[float],
    priorities: List[float],
    unroll_step: int,
) -> Iterable[Transition]:
    """Turn a list of episode history from t=0 to t=T, where T is terminal into a list of structured transition object,
    this is only for Atari and classic openAI Gym problems.

    Args:
        observations: a list of history environment observations.
        actions: a list of history actual actions taken in the environment.
        rewards: a list of history reward received from the environment.
        pi_probs: a list of history policy probabilities from the MCTS search result.
        values: a list of n-step target value.
        priorities: a list of priorities for each transition.
        unroll_step: number of unroll steps during traning.

    Returns:
        yeilds tuple of structured Transition object and the associated priority for the specific transition.

    """

    T = len(observations)

    # States past the end of games are treated as absorbing states.
    if len(actions) == T:
        actions += [0] * unroll_step
    if len(rewards) == T:
        rewards += [0] * unroll_step
    if len(values) == T:
        values += [0] * unroll_step
    if len(pi_probs) == T:
        # Uniform policy
        uniform_policy = np.ones_like(pi_probs[-1]) / len(pi_probs[-1])
        pi_probs += [uniform_policy] * unroll_step

    assert len(actions) == len(rewards) == len(values) == len(pi_probs) == T + unroll_step

    for t in range(T):
        end_index = t + unroll_step
        stacked_action = np.array(actions[t:end_index], dtype=np.int8)
        stacked_reward = np.array(rewards[t:end_index], dtype=np.float32)
        stacked_value = np.array(values[t:end_index], dtype=np.float32)
        stacked_pi_prob = np.array(pi_probs[t:end_index], dtype=np.float32)

        yield (
            Transition(
                state=observations[t],
                action=stacked_action,
                reward=stacked_reward,
                value=stacked_value,
                pi_prob=stacked_pi_prob,
            ),
            priorities[t],
        )


def init_absl_logging():
    """Initialize absl.logging when run the process without app.run()"""
    logging._warn_preinit_stderr = 0  # pylint: disable=protected-access
    logging.set_verbosity(logging.INFO)
    logging.use_absl_handler()


def handle_exit_signal():
    """Listen to exit signal like ctrl-c or kill from os and try to exit the process forcefully."""

    def shutdown(signal_code, frame):
        del frame
        logging.info(
            f'Received signal {signal_code}: terminating process...',
        )
        sys.exit(128 + signal_code)

    # Listen to signals to exit process.
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)


def get_time_stamp(as_file_name: bool = False) -> str:
    t = time.localtime()
    if as_file_name:
        timestamp = time.strftime('%Y%m%d_%H%M%S', t)
    else:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', t)
    return timestamp


def create_checkpoint(state_to_save: Mapping[Text, Any], ckpt_file: str) -> None:
    torch.save(state_to_save, ckpt_file)


def load_checkpoint(ckpt_file: str, device: torch.device) -> Mapping[Text, Any]:
    return torch.load(ckpt_file, map_location=torch.device(device))


def save_to_file(obj: Any, file_name: str) -> None:
    """Save object to file."""
    pickle.dump(obj, open(file_name, 'wb'))


def load_from_file(file_name: str) -> Any:
    """Load object from file."""
    return pickle.load(open(file_name, 'rb'))


def disable_auto_grad(network: torch.nn.Module) -> None:
    for p in network.parameters():
        p.requires_grad = False
