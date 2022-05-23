from absl import logging
from typing import Callable, Iterable, List, Tuple, Mapping, Text, Any
from pathlib import Path
import shutil
import os
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
from torch.utils.tensorboard import SummaryWriter

import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

from muzero.network import MuZeroNet
from muzero.replay import Transition, TransitionStructure, PrioritizedReplay
from muzero.trackers import make_self_play_trackers, make_learner_trackers
from muzero.mcts import uct_search
from muzero.util import scalar_to_support, support_to_scalar


@torch.no_grad()
def run_self_play(
    rank: int,
    network: MuZeroNet,
    device: torch.device,
    env: gym.Env,
    data_queue: multiprocessing.Queue,
    train_steps_counter: multiprocessing.Value,
    checkpoint_files: List,
    num_simulations: int,
    n_step: int,
    unroll_step: int,
    discount: float,
    root_noise_alpha: float,
    # temperature: float,
    stop_event: multiprocessing.Event,
) -> None:
    """Run self-play for as long as needed, only stop if stop_event is set to True."""

    init_absl_logging()
    handle_exit_signal()
    logging.info(f'Start self-play actor {rank}')

    tb_log_dir = f'actor{rank}'
    trackers = make_self_play_trackers(tb_log_dir)
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

        if len(checkpoint_files) > 0:
            ckpt_file = checkpoint_files[-1]
            loaded_state = load_checkpoint(ckpt_file, device)
            network.load_state_dict(loaded_state['network'])

        # Play and record transitions.
        while not done:
            action, pi_prob, root_value, v_error = uct_search(
                state=obs,
                network=network,
                device=device,
                discount=discount,
                temperature=visit_temp_func(train_steps_counter.value),
                num_simulations=num_simulations,
                root_noise_alpha=root_noise_alpha,
            )

            next_obs, reward, done, _ = env.step(action)

            for tracker in trackers:
                tracker.step(reward, done)

            episode_trajectory.append((obs, action, reward, pi_prob, root_value, v_error))

            obs = next_obs

        game += 1
        if game % 100 == 0:
            logging.info(f'Self-play actor {rank} played {game} games')

        # Compute n_step target value and make unroll sequences.
        observations, actions, rewards, pi_probs, root_values, priorities = map(list, zip(*episode_trajectory))
        target_values = compute_n_step_target(rewards, root_values, n_step, discount)
        for transition, priority in make_unroll_sequence(
            observations, actions, rewards, pi_probs, target_values, priorities, unroll_step
        ):
            data_queue.put((transition, priority))

        del episode_trajectory[:]

    logging.info(f'Stop self-play actor {rank}')


def run_training(  # noqa: C901
    network: MuZeroNet,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,
    device: torch.device,
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

    trackers = make_learner_trackers('learner')
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
        loss, priorities = calc_loss(network, device, transitions, weights)
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
            ckpt_file = ckpt_dir / f'train_steps_{train_steps_counter.value}'
            create_checkpoint(state_to_save, ckpt_file)

            if len(checkpoint_files) > 0:
                checkpoint_files.pop(0)

            checkpoint_files.append(ckpt_file)

            del state_to_save

        # Wait for sometime before start training on next batch.
        if delay is not None and delay > 0 and train_steps_counter.value > 1:
            time.sleep(delay)

    time.sleep(30)
    stop_event.set()


def run_data_collector(
    data_queue: multiprocessing.SimpleQueue,
    replay: PrioritizedReplay,
    save_frequency: int,
    save_dir: str,
    stop_event: multiprocessing.Event,
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
                save_file = save_samples_dir / f'replay_{replay.size}_{get_time_stamp(True)}'
                save_to_file(replay.get_state(), save_file)
                logging.info(f"Replay samples saved to '{save_file}'")
        except queue.Empty:
            pass
        except EOFError:
            pass


def calc_loss(network: MuZeroNet, device: torch.device, transitions: Transition, weights: torch.Tensor) -> torch.Tensor:
    # [B, state_shape]
    state = torch.from_numpy(transitions.state).to(device=device, dtype=torch.float32, non_blocking=True)
    # [B, T]
    action = torch.from_numpy(transitions.action).to(device=device, dtype=torch.long, non_blocking=True)
    value = torch.from_numpy(transitions.value).to(device=device, dtype=torch.float32, non_blocking=True)
    reward = torch.from_numpy(transitions.reward).to(device=device, dtype=torch.float32, non_blocking=True)
    # [B, T, num_actions]
    target_pi_prob = torch.from_numpy(transitions.pi_prob).to(device=device, dtype=torch.float32, non_blocking=True)

    # Compute losses
    B, T = action.shape
    reward_loss, value_loss, policy_loss = (0, 0, 0)
    loss_scale = 1.0 / T

    hidden_state = network.represent(state)
    pi_logits, value_logits = network.prediction(hidden_state)

    with torch.no_grad():
        # Priorities.
        pred_value_scalar = support_to_scalar(value_logits, network.value_supports.detach()).squeeze(1)
        priorities = torch.abs(pred_value_scalar - value[:, 0]).cpu().numpy()

        target_value = scalar_to_support(value, network.value_support_size)
        target_reward = scalar_to_support(reward, network.reward_support_size)

    # First step value and policy losses.
    value_loss += scalar_loss(value_logits, target_value[:, 0])
    policy_loss += scalar_loss(pi_logits, target_pi_prob[:, 0])

    # Unroll K-1 steps, skip first step.
    for t in range(1, T):
        network_output = network.unroll_dynamics_and_prediction(hidden_state, action[:, t].unsqueeze(1))
        hidden_state = network_output.hidden_state
        hidden_state.register_hook(lambda grad: grad * 0.5)

        # value_loss += F.cross_entropy(network_output.value_logits, target_value[:, t], reduction='none')
        # reward_loss += F.cross_entropy(network_output.reward_logits, target_reward[:, t], reduction='none')
        # policy_loss += F.cross_entropy(network_output.pi_logits, pi_prob[:, t], reduction='none')

        value_loss += scalar_loss(network_output.value_logits, target_value[:, t])
        reward_loss += scalar_loss(network_output.reward_logits, target_reward[:, t])
        policy_loss += scalar_loss(network_output.pi_logits, target_pi_prob[:, t])

    reward_loss = reward_loss * weights.detach()
    value_loss = value_loss * weights.detach()
    policy_loss = policy_loss * weights.detach()
    loss = torch.mean(reward_loss + value_loss + policy_loss)
    loss.register_hook(lambda grad: grad * loss_scale)

    return loss, priorities


def scalar_loss(prediction: torch.Tensor, target: torch.Tensor, mse: bool = False) -> torch.Tensor:

    assert prediction.shape == target.shape
    assert len(prediction.shape) == 2

    if mse:
        # [B]
        return 0.5 * torch.square((prediction - target))

    # [B]
    return torch.sum(-target * F.log_softmax(prediction, dim=1), dim=1)


def visit_temp_func(train_steps) -> float:
    # if is_board_game:
    #     if env_steps < 30:
    #         return 1.0
    #     else:
    #         return 0.1  # Play according to the max.
    # else:
    if train_steps < 500e3:
        return 1.0
    elif train_steps < 750e3:
        return 0.5
    else:
        return 0.25


def compute_n_step_target(
    input_rewards: List[float], input_root_values: List[float], n_step: int, discount: float
) -> List[float]:
    """Compute n-step target value for Atari and classic openAI Gym games."""

    T = len(input_rewards)

    rewards = copy.deepcopy(input_rewards)
    root_values = copy.deepcopy(input_root_values)

    # Padding zeros at the end of trajectory for easy computation.
    rewards += [0] * n_step
    root_values += [0] * n_step

    target_values = []

    for t in range(T):
        bootstrap_index = t + n_step
        n_step_target = sum([r * discount**i for i, r in enumerate(rewards[t:bootstrap_index])])

        # Add MCTS node root value for bootstrap.
        n_step_target += root_values[bootstrap_index]

        target_values.append(n_step_target)

    return target_values


def make_unroll_sequence(observations, actions, rewards, pi_probs, values, priorities, unroll_step) -> Iterable[Transition]:
    T = len(observations)

    # States past the end of games are treated as absorbing states.
    actions += [0] * unroll_step
    rewards += [0] * unroll_step
    values += [0] * unroll_step

    # uniform policy
    uniform_policy = np.ones_like(pi_probs[-1]) / len(pi_probs[-1])
    pi_probs += [uniform_policy] * unroll_step

    for t in range(T):
        end_index = t + unroll_step
        stacked_action = np.array(actions[t:end_index], dtype=np.int8)
        stacked_reward = np.array(rewards[t:end_index], dtype=np.float32)
        stacked_value = np.array(values[t:end_index], dtype=np.float32)
        stacked_pi_prob = np.array(pi_probs[t:end_index], dtype=np.float32)

        stacked_done = np.array([False if i < T else True for i in range(end_index, end_index + unroll_step)])

        yield (
            Transition(
                state=observations[t],
                action=stacked_action,
                reward=stacked_reward,
                value=stacked_value,
                pi_prob=stacked_pi_prob,
                done=stacked_done,
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
