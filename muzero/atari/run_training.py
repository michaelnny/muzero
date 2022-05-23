"""Runs MuZero self-play training pipeline on Atari game.

From the paper "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
https://arxiv.org/abs//1712.01815


IMPORTANT NOTE:
In order to run this, one needs to run large amount of actors (like 256 in MuZero paper) to generate self-play samples,
as training one batch size 128 only takes 20-30 seconds.
But self-play one game could take 3-5 minutes (in the early stage one game only have 50-80 samples).
In the case we run training on single machine with 8-16 actors,
the network could easily overfitting to existing samples while fail to converge to even suboptimal policy.

One hack is to add some delay before start traning on next batch, to wait for the actors to generate some more training samples,
this will slow down the overall training progress, but it should work in theory.
The downside is we also have to 'tune' the delay hyper-parameters.


nohup python3 -m muzero.atari.run_training --num_actors=6 --train_delay=0.25 --batch_size=32 --load_samples_file=samples/atari/replay_10000_20220518_133013 &

"""
from absl import app
from absl import flags
from absl import logging
import os
import multiprocessing
import threading
import torch
from torch.optim.lr_scheduler import MultiStepLR

from muzero.network import MuZeroConvNet
from muzero.replay import PrioritizedReplay
from muzero.gym_env import create_atari_environment
from muzero.pipeline import run_self_play, run_training, run_evaluation, run_data_collector, load_checkpoint, load_from_file

FLAGS = flags.FLAGS
flags.DEFINE_string("environment_name", 'Breakout', "Classic problem like Breakout, Pong")
flags.DEFINE_integer('environment_height', 96, 'Environment frame screen height.')
flags.DEFINE_integer('environment_width', 96, 'Environment frame screen width.')
flags.DEFINE_integer("stack_history", 4, "Stack previous states.")
flags.DEFINE_bool("gray_scale", True, "Gray scale observation image.")

flags.DEFINE_integer('num_res_blocks', 6, 'Number of res-blocks in the dynamics function.')
flags.DEFINE_integer('num_planes', 128, 'Number of planes for Conv2d layers in the model.')
flags.DEFINE_integer('support_size', 101, 'Value and reward scalar projection support size.')

flags.DEFINE_float('learning_rate', 0.05, 'Learning rate.')
flags.DEFINE_float('l2_decay', 0.0001, 'Adam L2 regularization.')
flags.DEFINE_multi_integer('lr_boundaries', [350000], 'The number of steps at which the learning rate will decay.')

flags.DEFINE_float('discount', 0.997, 'Gamma discount.')
flags.DEFINE_integer('n_step', 10, 'Value n-step bootstrap.')
flags.DEFINE_integer('unroll_k', 5, 'Unroll dynamics and prediction functions for K steps during training.')

flags.DEFINE_integer(
    'replay_capacity', 10000, 'Maximum replay size, note one sample contains unroll_k sequence length unroll.'
)  # 50000
flags.DEFINE_integer('min_replay_size', 5000, 'Minimum replay size before learning starts.')
flags.DEFINE_integer('batch_size', 16, 'Sample batch size when do learning.')  # 32

flags.DEFINE_integer('num_train_steps', 1000000, 'Number of network updates per iteration.')

flags.DEFINE_integer('num_actors', 8, 'Number of self-play actor processes.')
flags.DEFINE_integer('num_simulations', 50, 'Number of simulations per MCTS search, per agent environment time step.')
flags.DEFINE_float(
    'root_noise_alpha', 0.25, 'Alph for dirichlet noise to MCTS root node prior probabilities to encourage exploration.'
)

flags.DEFINE_float(
    'train_delay',
    0.0,
    'Delay (in seconds) before training on next batch samples, if training on GPU, using large value (like 0.75, 1.0, 1.5).',
)
flags.DEFINE_integer('seed', 1, 'Seed the runtime.')

flags.DEFINE_integer('checkpoint_frequency', 1000, 'The frequency (in training step) to create new checkpoint.')
flags.DEFINE_string('checkpoint_dir', 'checkpoints/atari', 'Path for checkpoint file.')
flags.DEFINE_string(
    'load_checkpoint_file',
    '',
    'Load the checkpoint from file.',
)

flags.DEFINE_integer(
    'samples_save_frequency',
    10000,
    'The frequency (measured in number added in replay) to save self-play samples in replay.',
)
flags.DEFINE_string('samples_save_dir', 'samples/atari', 'Path for save self-play samples in replay to file.')
flags.DEFINE_string('load_samples_file', '', 'Load the replay samples from file.')
flags.DEFINE_string('train_csv_file', 'logs/train_atari.csv', 'A csv file contains training statistics.')
flags.DEFINE_string('eval_csv_file', 'logs/eval_atari.csv', 'A csv file contains training statistics.')


def main(argv):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    runtime_device = torch.device(device)

    evaluation_env = create_atari_environment(
        FLAGS.environment_name,
        FLAGS.seed + 1,
        FLAGS.stack_history,
        FLAGS.environment_height,
        FLAGS.environment_width,
        grayscale=FLAGS.gray_scale,
    )
    self_play_envs = [
        create_atari_environment(
            FLAGS.environment_name,
            FLAGS.seed + i**2,
            FLAGS.stack_history,
            FLAGS.environment_height,
            FLAGS.environment_width,
            grayscale=FLAGS.gray_scale,
        )
        for i in range(FLAGS.num_actors)
    ]

    input_shape = self_play_envs[0].observation_space.shape
    num_actions = self_play_envs[0].action_space.n

    network = MuZeroConvNet(input_shape, num_actions, FLAGS.num_res_blocks, FLAGS.num_planes, FLAGS.support_size)
    optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.l2_decay)
    lr_scheduler = MultiStepLR(optimizer, milestones=FLAGS.lr_boundaries, gamma=0.1)

    actor_network = MuZeroConvNet(input_shape, num_actions, FLAGS.num_res_blocks, FLAGS.num_planes, FLAGS.support_size)
    actor_network.share_memory()

    new_checkpoint_network = MuZeroConvNet(
        input_shape, num_actions, FLAGS.num_res_blocks, FLAGS.num_planes, FLAGS.support_size
    )

    replay = PrioritizedReplay(FLAGS.replay_capacity)

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
        args=(
            data_queue,
            replay,
            FLAGS.unroll_k,
            FLAGS.samples_save_frequency,
            FLAGS.samples_save_dir,
            stop_event,
        ),
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
            FLAGS.checkpoint_frequency,
            FLAGS.checkpoint_dir,
            checkpoint_files,
            FLAGS.train_csv_file,
            stop_event,
            FLAGS.train_delay,
        ),
    )
    learner.start()

    # Start evaluation loop on a seperate process.
    evaluator = multiprocessing.Process(
        target=run_evaluation,
        args=(
            new_checkpoint_network,
            runtime_device,
            evaluation_env,
            0.35,
            FLAGS.num_simulations,
            checkpoint_files,
            FLAGS.eval_csv_file,
            stop_event,
            FLAGS.discount,
            50,
        ),
    )
    evaluator.start()

    # Start self-play processes.
    actors = []
    for i in range(FLAGS.num_actors):
        actor = multiprocessing.Process(
            target=run_self_play,
            args=(
                i,
                actor_network,
                'cpu',
                self_play_envs[i],
                data_queue,
                train_steps_counter,
                FLAGS.num_simulations,
                FLAGS.n_step,
                FLAGS.unroll_k,
                FLAGS.discount,
                stop_event,
                FLAGS.root_noise_alpha,
                50,
            ),
        )
        actor.start()
        actors.append(actor)

    for actor in actors:
        actor.join()
        actor.close()

    evaluator.join()
    learner.join()
    data_collector.join()


if __name__ == '__main__':
    # Set multiprocessing start mode
    multiprocessing.set_start_method('spawn')
    app.run(main)
