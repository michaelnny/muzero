# Copyright 2022 Michael Hu. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs MuZero self-play training pipeline on free-style Gomoku game.
"""
from absl import app
from absl import flags
from absl import logging
import os
import multiprocessing
import threading
import torch
from torch.optim.lr_scheduler import MultiStepLR

from muzero.games.gomoku import GomokuEnv
from muzero.network import MuZeroBoardGameNet
from muzero.replay import PrioritizedReplay
from muzero.core import make_gomoku_config
from muzero.pipeline import (
    run_self_play,
    run_training,
    run_data_collector,
    run_board_game_evaluator,
    load_checkpoint,
    load_from_file,
)


FLAGS = flags.FLAGS
flags.DEFINE_integer('board_size', 9, 'Board size for Gomoku.')
flags.DEFINE_integer('num_to_win', 5, 'Number in a row to win.')
flags.DEFINE_integer('stack_history', 4, 'Stack previous states.')
flags.DEFINE_integer('seed', 1, 'Seed the runtime.')
flags.DEFINE_float('initial_elo', 0.0, 'Initial elo rating, for evaluation agent performance only.')
flags.DEFINE_string('checkpoint_dir', 'checkpoints/gomoku', 'Path for checkpoint file.')
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
flags.DEFINE_string('samples_save_dir', 'samples/gomoku', 'Path for save self-play samples in replay to file.')
flags.DEFINE_string('load_samples_file', '', 'Load the replay samples from file.')
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')


def main(argv):
    """Trains MuZero agent on Gomoku board game."""
    del argv

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    runtime_device = torch.device(device)

    self_play_envs = [
        GomokuEnv(board_size=FLAGS.board_size, num_to_win=FLAGS.num_to_win, stack_history=FLAGS.stack_history)
        for i in range(FLAGS.num_actors)
    ]
    eval_env = GomokuEnv(board_size=FLAGS.board_size, num_to_win=FLAGS.num_to_win, stack_history=FLAGS.stack_history)

    input_shape = self_play_envs[0].observation_space.shape
    num_actions = self_play_envs[0].action_space.n

    tag = 'Gomoku'
    if FLAGS.tag is not None and FLAGS.tag != '':
        tag = f'{tag}_{FLAGS.tag}'

    config = make_gomoku_config()

    network = MuZeroBoardGameNet(input_shape, num_actions, config.num_res_blocks, config.num_planes)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)
    lr_scheduler = MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=config.lr_decay_rate)

    actor_network = MuZeroBoardGameNet(input_shape, num_actions, config.num_res_blocks, config.num_planes)
    actor_network.share_memory()

    # For evaluation only.
    old_ckpt_network = MuZeroBoardGameNet(input_shape, num_actions, config.num_res_blocks, config.num_planes)
    new_ckpt_network = MuZeroBoardGameNet(input_shape, num_actions, config.num_res_blocks, config.num_planes)

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
        old_ckpt_network.load_state_dict(loaded_state['network'])

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

    # Start evaluation loop on a seperate process.
    evaluator = multiprocessing.Process(
        target=run_board_game_evaluator,
        args=(
            config,
            old_ckpt_network,
            new_ckpt_network,
            runtime_device,
            eval_env,
            0.0,
            checkpoint_files,
            stop_event,
            FLAGS.initial_elo,
            tag,
        ),
    )
    evaluator.start()

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
    evaluator.join()


if __name__ == '__main__':
    # Set multiprocessing start mode
    multiprocessing.set_start_method('spawn')
    app.run(main)
