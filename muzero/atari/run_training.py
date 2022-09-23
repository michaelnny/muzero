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
"""Runs MuZero self-play training pipeline on Atari game."""
from absl import app
from absl import flags
from absl import logging
import os
import multiprocessing
import threading
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR

from muzero.network import MuZeroAtariNet
from muzero.replay import PrioritizedReplay
from muzero.config import make_atari_config
from muzero.gym_env import create_atari_environment
from muzero.pipeline import run_self_play, run_training, run_data_collector, run_evaluator, load_checkpoint, load_from_file

FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'Pong', 'Classic problem like Breakout, Pong')
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Number of frames to skip.')
flags.DEFINE_integer('environment_frame_stack', 8, 'Number of frames to stack.')
flags.DEFINE_integer('max_episode_steps', 108000, 'Maximum steps (before frame skip) per episode.')
flags.DEFINE_bool('gray_scale', True, 'Gray scale observation image.')
flags.DEFINE_bool('clip_reward', True, 'Clip reward in the range [-1, 1], default on.')
flags.DEFINE_bool('done_on_life_loss', True, 'End of game if loss a life, default on.')
flags.DEFINE_integer('num_actors', 6, 'Number of self-play actor processes.')

flags.DEFINE_integer('num_training_steps', int(10e6), 'Number of traning steps.')
flags.DEFINE_integer('batch_size', 128, 'Batch size for traning.')
flags.DEFINE_integer('replay_capacity', int(1e6), 'Maximum replay size.')
flags.DEFINE_integer('min_replay_size', 10000, 'Minimum replay size before start to do traning.')
flags.DEFINE_float(
    'priority_exponent', 0.0, 'Priotiry exponent used in prioritized replay, 0 means using uniform random replay.'
)
flags.DEFINE_float('importance_sampling_exponent', 0.0, 'Importance sampling exponent value.')

flags.DEFINE_integer('seed', 1, 'Seed the runtime.')
flags.DEFINE_bool('use_tensorboard', True, 'Monitor performance with Tensorboard, default on.')
flags.DEFINE_bool('clip_grad', True, 'Clip gradient, default off.')

flags.DEFINE_string('checkpoint_dir', 'checkpoints', 'Path for save checkpoint files.')
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
flags.DEFINE_string('samples_save_dir', 'samples', 'Path for save self-play samples in replay to file.')
flags.DEFINE_string('load_samples_file', '', 'Load the replay samples from file.')
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')


def main(argv):
    """Trains MuZero agent on Atari games."""
    del argv

    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def environment_builder():
        return create_atari_environment(
            env_name=FLAGS.environment_name,
            screen_height=FLAGS.environment_height,
            screen_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            max_episode_steps=FLAGS.max_episode_steps,
            seed=random_state.randint(1, 2**32),
            noop_max=30,
            terminal_on_life_loss=False,
            clip_reward=False,
        )

    self_play_envs = [environment_builder() for _ in range(FLAGS.num_actors)]

    eval_env = environment_builder()

    input_shape = self_play_envs[0].observation_space.shape
    num_actions = self_play_envs[0].action_space.n

    tag = self_play_envs[0].spec.id
    if FLAGS.tag is not None and FLAGS.tag != '':
        tag = f'{tag}_{FLAGS.tag}'

    config = make_atari_config(
        num_training_steps=FLAGS.num_training_steps,
        batch_size=FLAGS.batch_size,
        min_replay_size=FLAGS.min_replay_size,
        use_tensorboard=FLAGS.use_tensorboard,
        clip_grad=FLAGS.clip_grad,
    )

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
    new_ckpt_network = MuZeroAtariNet(
        input_shape,
        num_actions,
        config.num_res_blocks,
        config.num_planes,
        config.value_support_size,
        config.reward_support_size,
    )

    replay = PrioritizedReplay(
        FLAGS.replay_capacity,
        FLAGS.priority_exponent,
        FLAGS.importance_sampling_exponent,
        random_state,
    )

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

    # Start evaluation loop on a seperate process.
    evaluator = multiprocessing.Process(
        target=run_evaluator,
        args=(
            config,
            new_ckpt_network,
            runtime_device,
            eval_env,
            0.0,
            checkpoint_files,
            stop_event,
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
