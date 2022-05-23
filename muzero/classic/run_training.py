"""Runs MuZero self-play training pipeline on Atari game.

From the paper "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
https://arxiv.org/abs//1712.01815

"""
from absl import app
from absl import flags
from absl import logging
import os
import multiprocessing
import threading
import torch
from torch.optim.lr_scheduler import MultiStepLR

from muzero.network import MuZeroMLPNet
from muzero.replay import PrioritizedReplay
from muzero.gym_env import create_classic_environment
from muzero.pipeline import run_self_play, run_training, run_data_collector, load_checkpoint, load_from_file


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "environment_name", 'LunarLander-v2', "Classic problem like 'CartPole-v1', 'LunarLander-v2', 'MountainCar-v0'"
)
flags.DEFINE_integer("stack_history", 4, "Stack previous states.")

flags.DEFINE_integer('num_planes', 256, 'Number of planes for Conv2d layers in the model.')
flags.DEFINE_integer('value_support_size', 41, 'Value scalar projection support size.')
flags.DEFINE_integer('reward_support_size', 21, 'Reward scalar projection support size.')
flags.DEFINE_integer('hidden_size', 64, 'Hidden state vector size.')

flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate.')
flags.DEFINE_float('learning_rate_decay', 0.1, 'Adam learning rate decay rate.')
flags.DEFINE_multi_integer('lr_boundaries', [500000], 'The number of steps at which the learning rate will decay.')
flags.DEFINE_float('l2_decay', 0.0001, 'Adam L2 regularization.')

flags.DEFINE_bool('clip_grad', False, 'Clip gradients, default on.')
flags.DEFINE_float('max_grad_norm', 40.0, 'Max gradients norm when do gradients clip.')

flags.DEFINE_float('discount', 0.997, 'Gamma discount.')
flags.DEFINE_integer('n_step', 10, 'Value n-step bootstrap.')
flags.DEFINE_integer('unroll_step', 5, 'Unroll dynamics and prediction functions for K steps during training.')

flags.DEFINE_integer('replay_capacity', 200000, 'Maximum replay size.')
flags.DEFINE_integer('min_replay_size', 20000, 'Minimum replay size before learning starts.')
flags.DEFINE_integer('batch_size', 128, 'Sample batch size when do learning.')

flags.DEFINE_integer('num_train_steps', 1000000, 'Number of network updates per iteration.')

flags.DEFINE_integer('num_actors', 8, 'Number of self-play actor processes.')
flags.DEFINE_integer('num_simulations', 50, 'Number of simulations per MCTS search, per agent environment time step.')
flags.DEFINE_float(
    'root_noise_alpha', 0.25, 'Alph for dirichlet noise to MCTS root node prior probabilities to encourage exploration.'
)

flags.DEFINE_float(
    'train_delay',
    0,
    'Delay (in seconds) before training on next batch samples, if training on GPU, using large value (like 0.75, 1.0, 1.5).',
)
flags.DEFINE_integer('seed', 1, 'Seed the runtime.')

flags.DEFINE_integer('checkpoint_frequency', 1000, 'The frequency (in training step) to create new checkpoint.')
flags.DEFINE_string('checkpoint_dir', 'checkpoints/classic', 'Path for checkpoint file.')
flags.DEFINE_string(
    'load_checkpoint_file',
    '',
    'Load the checkpoint from file.',
)

flags.DEFINE_integer(
    'samples_save_frequency',
    100000,
    'The frequency (measured in number added in replay) to save self-play samples in replay.',
)
flags.DEFINE_string('samples_save_dir', 'samples/classic', 'Path for save self-play samples in replay to file.')
flags.DEFINE_string('load_samples_file', '', 'Load the replay samples from file.')


def main(argv):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    runtime_device = torch.device(device)

    self_play_envs = [
        create_classic_environment(FLAGS.environment_name, FLAGS.seed + i**2, FLAGS.stack_history)
        for i in range(FLAGS.num_actors)
    ]

    input_shape = self_play_envs[0].observation_space.shape
    num_actions = self_play_envs[0].action_space.n

    network = MuZeroMLPNet(
        input_shape, num_actions, FLAGS.num_planes, FLAGS.value_support_size, FLAGS.reward_support_size, FLAGS.hidden_size
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.l2_decay)
    lr_scheduler = MultiStepLR(optimizer, milestones=FLAGS.lr_boundaries, gamma=FLAGS.learning_rate_decay)

    actor_network = MuZeroMLPNet(
        input_shape, num_actions, FLAGS.num_planes, FLAGS.value_support_size, FLAGS.reward_support_size, FLAGS.hidden_size
    )
    actor_network.share_memory()

    replay = PrioritizedReplay(FLAGS.replay_capacity, priority_exponent=1.0, importance_sampling_exponent=1.0)

    # Train loop use the stop_event to signaling other parties to stop running the pipeline.
    stop_event = multiprocessing.Event()
    # Transfer samples from self-play process to training process.
    data_queue = multiprocessing.SimpleQueue()
    # A shared list to store most recent new checkpoint files.
    manager = multiprocessing.Manager()
    checkpoint_files = manager.list()

    train_steps_counter = multiprocessing.Value('i', 0)

    # Load states from checkpoint to resume training.
    if FLAGS.load_checkpoint_file is not None and os.path.isfile(FLAGS.load_checkpoint_file):
        loaded_state = load_checkpoint(FLAGS.load_checkpoint_file, 'cpu')
        network.load_state_dict(loaded_state['network'])
        optimizer.load_state_dict(loaded_state['optimizer'])
        lr_scheduler.load_state_dict(loaded_state['lr_scheduler'])
        train_steps_counter.value = loaded_state['train_steps']

        actor_network.load_state_dict(loaded_state['network'])

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
        args=(data_queue, replay, FLAGS.samples_save_frequency, FLAGS.samples_save_dir, stop_event),
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
        ),
    )
    learner.start()

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
                checkpoint_files,
                FLAGS.num_simulations,
                FLAGS.n_step,
                FLAGS.unroll_step,
                FLAGS.discount,
                FLAGS.root_noise_alpha,
                stop_event,
            ),
        )
        actor.start()
        actors.append(actor)

    for actor in actors:
        actor.join()
        actor.close()

    learner.join()
    data_collector.join()


if __name__ == '__main__':
    # Set multiprocessing start mode
    multiprocessing.set_start_method('spawn')
    app.run(main)
