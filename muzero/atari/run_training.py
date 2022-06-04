"""Runs MuZero self-play training pipeline on Atari game.
"""
from absl import app
from absl import flags
from absl import logging
import os
import multiprocessing
import threading
import torch
from torch.optim.lr_scheduler import MultiStepLR

from muzero.network import MuZeroAtariNet
from muzero.replay import PrioritizedReplay
from muzero.core import make_atari_config
from muzero.gym_env import create_atari_environment
from muzero.pipeline import run_self_play, run_training, run_data_collector, load_checkpoint, load_from_file

FLAGS = flags.FLAGS
flags.DEFINE_string("environment_name", 'BreakoutNoFrameskip-v4', "Classic problem like Breakout, Pong")
flags.DEFINE_integer('screen_size', 84, 'Environment frame screen height.')
flags.DEFINE_integer("stack_history", 4, "Stack previous states.")
flags.DEFINE_integer("frame_skip", 4, "Skip n frames.")
flags.DEFINE_bool("gray_scale", True, "Gray scale observation image.")

flags.DEFINE_integer('num_actors', 6, 'Number of self-play actor processes.')

flags.DEFINE_integer('seed', 1, 'Seed the runtime.')

flags.DEFINE_string('checkpoint_dir', 'checkpoints/atari', 'Path for checkpoint file.')
flags.DEFINE_string(
    'load_checkpoint_file',
    '',
    'Load the checkpoint from file.',
)

flags.DEFINE_integer(
    'samples_save_frequency',
    -1,
    'The frequency (measured in number added in replay) to save self-play samples in replay, default -1 do not save.',
)
flags.DEFINE_string('samples_save_dir', 'samples/atari', 'Path for save self-play samples in replay to file.')
flags.DEFINE_string('load_samples_file', '', 'Load the replay samples from file.')
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')


def main(argv):
    """Trains MuZero agent on Atari games."""
    del argv

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    runtime_device = torch.device(device)

    self_play_envs = [
        create_atari_environment(
            FLAGS.environment_name,
            FLAGS.seed + i**2,
            FLAGS.stack_history,
            FLAGS.frame_skip,
            FLAGS.screen_size,
            grayscale=FLAGS.gray_scale,
        )
        for i in range(FLAGS.num_actors)
    ]

    input_shape = self_play_envs[0].observation_space.shape
    num_actions = self_play_envs[0].action_space.n

    tag = self_play_envs[0].spec.id
    if FLAGS.tag is not None and FLAGS.tag != '':
        tag = f'{tag}_{FLAGS.tag}'

    config = make_atari_config()

    network = MuZeroAtariNet(
        input_shape,
        num_actions,
        config.num_res_blocks,
        config.num_planes,
        config.value_support_size,
        config.reward_support_size,
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)
    lr_scheduler = MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=config.lr_decay_rate)

    actor_network = MuZeroAtariNet(
        input_shape,
        num_actions,
        config.num_res_blocks,
        config.num_planes,
        config.value_support_size,
        config.reward_support_size,
    )
    actor_network.share_memory()

    replay = PrioritizedReplay(config.replay_capacity, config.priority_exponent, config.importance_sampling_exponent)

    # Use the stop_event to signaling actors to stop running.
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
        args=(data_queue, replay, FLAGS.samples_save_frequency, FLAGS.samples_save_dir, tag),
    )
    data_collector.start()

    # Start the main training loop on a new thread.
    learner = threading.Thread(
        target=run_training,
        args=(
            config,
            network,
            optimizer,
            lr_scheduler,
            runtime_device,
            actor_network,
            replay,
            data_queue,
            train_steps_counter,
            FLAGS.checkpoint_dir,
            checkpoint_files,
            stop_event,
            tag,
        ),
    )
    learner.start()

    # # Start self-play processes.
    actors = []
    for i in range(FLAGS.num_actors):
        actor = multiprocessing.Process(
            target=run_self_play,
            args=(
                config,
                i,
                actor_network,
                runtime_device,
                self_play_envs[i],
                data_queue,
                train_steps_counter,
                stop_event,
                tag,
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
