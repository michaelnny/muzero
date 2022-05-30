"""Runs MuZero self-play training pipeline on Tic-Tac-Toe game.

"""
from absl import app
from absl import flags
from absl import logging
import os
import multiprocessing
import threading
import torch
from torch.optim.lr_scheduler import MultiStepLR

from muzero.games.tictactoe import TicTacToeEnv
from muzero.network import MuZeroMLPNet, MuZeroBoardGameNet
from muzero.replay import PrioritizedReplay
from muzero.pipeline import run_self_play, run_evaluation, run_training, run_data_collector, load_checkpoint, load_from_file

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_res_blocks', 2, 'Number of res-blocks in the representation, dynamics, and prediction functions.')
flags.DEFINE_integer('num_planes', 512, 'Number of planes for Conv2d layers and (hidden units) FC layers in the model.')

flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate.')
flags.DEFINE_float('learning_rate_decay', 0.1, 'Adam learning rate decay rate.')
flags.DEFINE_multi_integer('lr_boundaries', [100000], 'The number of steps at which the learning rate will decay.')
flags.DEFINE_float('l2_decay', 0.0001, 'Adam L2 regularization.')

flags.DEFINE_bool('clip_grad', False, 'Clip gradients, default on.')
flags.DEFINE_float('max_grad_norm', 40.0, 'Max gradients norm when do gradients clip.')

flags.DEFINE_float('discount', 1.0, 'Gamma discount.')
flags.DEFINE_integer('n_step', 10, 'Value n-step bootstrap.')
flags.DEFINE_integer('unroll_step', 5, 'Unroll dynamics and prediction functions for K steps during training.')

flags.DEFINE_integer('replay_capacity', 50000, 'Maximum replay size.')
flags.DEFINE_integer('min_replay_size', 20000, 'Minimum replay size before learning starts.')
flags.DEFINE_integer('batch_size', 128, 'Sample batch size when do learning.')

flags.DEFINE_integer('num_train_steps', 50000, 'Number of training steps (measured in network updates).')

flags.DEFINE_integer('num_actors', 7, 'Number of self-play actor processes.')
flags.DEFINE_integer('num_simulations', 30, 'Number of simulations per MCTS search, per agent environment time step.')
flags.DEFINE_float(
    'root_noise_alpha', 0.25, 'Alph for dirichlet noise to MCTS root node prior probabilities to encourage exploration.'
)
flags.DEFINE_float('pb_c_base', 19652.0, 'PB C Base.')
flags.DEFINE_float('pb_c_init', 1.25, 'PB C Init.')

flags.DEFINE_float('min_bound', -1.0, 'Minimum value bound.')
flags.DEFINE_float('max_bound', 1.0, 'Maximum value bound.')

flags.DEFINE_float(
    'train_delay',
    0.0,
    'Delay (in seconds) before training on next batch samples, if training on GPU, using large value (like 0.75, 1.0, 1.5).',
)
flags.DEFINE_float(
    'initial_elo', 0.0, 'Initial elo rating, in case resume training, this should be the elo form last checkpoint.'
)

flags.DEFINE_integer('seed', 1, 'Seed the runtime.')

flags.DEFINE_integer('checkpoint_frequency', 1000, 'The frequency (in training step) to create new checkpoint.')
flags.DEFINE_string('checkpoint_dir', 'checkpoints/tictactoe', 'Path for checkpoint file.')
flags.DEFINE_string(
    'load_checkpoint_file',
    '',
    'Load the checkpoint from file.',
)

flags.DEFINE_integer(
    'samples_save_frequency',
    -1,
    'The frequency (measured in number added in replay) to save self-play samples in replay.',
)
flags.DEFINE_string('samples_save_dir', 'samples/tictactoe', 'Path for save self-play samples in replay to file.')
flags.DEFINE_string('load_samples_file', '', 'Load the replay samples from file.')
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')


def mcts_temp_func(env_steps: int, train_steps: int) -> float:
    """Board game MCTS temperature scheduler."""
    if env_steps < 6:
        return 1.0
    return 0.1


def main(argv):
    """Trains MuZero agent on Gomoku board game."""
    del argv

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    runtime_device = torch.device(device)

    self_play_envs = [TicTacToeEnv() for i in range(FLAGS.num_actors)]
    evaluation_env = TicTacToeEnv()

    input_shape = self_play_envs[0].observation_space.shape
    num_actions = self_play_envs[0].action_space.n

    tag = 'TicTacToe'
    if FLAGS.tag is not None and FLAGS.tag != '':
        tag = f'{tag}_{FLAGS.tag}'

    # network = MuZeroBoardGameNet(input_shape, num_actions, FLAGS.num_res_blocks, FLAGS.num_planes)
    network = MuZeroMLPNet(input_shape, num_actions, FLAGS.num_planes, 31, 31, 64)
    optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.l2_decay, eps=0.00015)
    lr_scheduler = MultiStepLR(optimizer, milestones=FLAGS.lr_boundaries, gamma=0.1)

    actor_network = MuZeroMLPNet(input_shape, num_actions, FLAGS.num_planes, 31, 31, 64)
    actor_network.share_memory()

    old_checkpoint_network = MuZeroMLPNet(input_shape, num_actions, FLAGS.num_planes, 31, 31, 64)
    new_checkpoint_network = MuZeroMLPNet(input_shape, num_actions, FLAGS.num_planes, 31, 31, 64)

    # By set priority_exponent=0, this becomes uniform replay
    replay = PrioritizedReplay(FLAGS.replay_capacity, priority_exponent=1.0, importance_sampling_exponent=1.0)

    # Train loop use the stop_event to signaling other parties to stop running the pipeline.
    stop_event = multiprocessing.Event()
    # Transfer samples from self-play process to training process.
    data_queue = multiprocessing.SimpleQueue()
    # A shared list to store most recent new checkpoint files.
    manager = multiprocessing.Manager()
    checkpoint_files = manager.list()

    # Shared training steps counter, so actors can adjust temperature used in MCTS.
    train_steps_counter = multiprocessing.Value('i', 0)

    # Load states from checkpoint to resume training.
    if FLAGS.load_checkpoint_file is not None and os.path.isfile(FLAGS.load_checkpoint_file):
        loaded_state = load_checkpoint(FLAGS.load_checkpoint_file, 'cpu')
        network.load_state_dict(loaded_state['network'])
        optimizer.load_state_dict(loaded_state['optimizer'])
        lr_scheduler.load_state_dict(loaded_state['lr_scheduler'])
        train_steps_counter.value = loaded_state['train_steps']

        actor_network.load_state_dict(loaded_state['network'])
        old_checkpoint_network.load_state_dict(loaded_state['network'])

        logging.info(f'Loaded state from checkpoint {FLAGS.load_checkpoint_file}')
        logging.info(f'Current state: train steps {train_steps_counter.value}, learing rate {lr_scheduler.get_last_lr()}')

    # Load replay samples
    if FLAGS.load_samples_file is not None and os.path.isfile(FLAGS.load_samples_file):
        try:
            replay.reset()
            replay_state = load_from_file(FLAGS.load_samples_file)
            replay.set_state(replay_state)
            logging.info(f"Loaded replay samples from file '{FLAGS.load_samples_file}'")
        except Exception:
            pass

    # Start to collect samples from self-play on a new thread.
    data_collector = threading.Thread(
        target=run_data_collector,
        args=(data_queue, replay, FLAGS.samples_save_frequency, FLAGS.samples_save_dir, stop_event, tag),
    )
    data_collector.start()

    # Start the main training loop on a new thread.
    learner = threading.Thread(
        target=run_training,
        args=(
            network,
            optimizer,
            lr_scheduler,
            runtime_device,
            actor_network,
            replay,
            FLAGS.min_replay_size,
            FLAGS.batch_size,
            FLAGS.num_train_steps,
            train_steps_counter,
            FLAGS.clip_grad,
            FLAGS.max_grad_norm,
            FLAGS.checkpoint_frequency,
            FLAGS.checkpoint_dir,
            checkpoint_files,
            stop_event,
            FLAGS.train_delay,
            tag,
            True,
        ),
    )
    learner.start()

    # # Start evaluation loop on a seperate process.
    # evaluator = multiprocessing.Process(
    #     target=run_evaluation,
    #     args=(
    #         old_checkpoint_network,
    #         new_checkpoint_network,
    #         runtime_device,
    #         evaluation_env,
    #         FLAGS.discount,
    #         FLAGS.pb_c_base,
    #         FLAGS.pb_c_init,
    #         0.0,
    #         0.1,
    #         FLAGS.num_simulations,
    #         FLAGS.min_bound,
    #         FLAGS.max_bound,
    #         checkpoint_files,
    #         stop_event,
    #         FLAGS.initial_elo,
    #         tag,
    #     ),
    # )
    # evaluator.start()

    # # Start self-play processes.
    actors = []
    for i in range(FLAGS.num_actors):
        actor = multiprocessing.Process(
            target=run_self_play,
            args=(
                i,
                actor_network,
                runtime_device,
                self_play_envs[i],
                data_queue,
                train_steps_counter,
                mcts_temp_func,
                FLAGS.num_simulations,
                FLAGS.n_step,
                FLAGS.unroll_step,
                FLAGS.discount,
                FLAGS.pb_c_base,
                FLAGS.pb_c_init,
                FLAGS.root_noise_alpha,
                FLAGS.min_bound,
                FLAGS.max_bound,
                stop_event,
                tag,
                True,
                True,
            ),
        )
        actor.start()
        actors.append(actor)

    for actor in actors:
        actor.join()
        actor.close()

    learner.join()
    data_collector.join()
    # evaluator.join()


if __name__ == '__main__':
    # Set multiprocessing start mode
    multiprocessing.set_start_method('spawn')
    app.run(main)
