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
from absl import app
from absl import flags
from absl import logging
import os
from pathlib import Path
import gym
import torch

from muzero.network import MuZeroAtariNet
from muzero.pipeline import load_checkpoint
from muzero.mcts import uct_search
from muzero.config import make_atari_config
from muzero.gym_env import create_atari_environment

FLAGS = flags.FLAGS
flags.DEFINE_string("environment_name", 'BreakoutNoFrameskip-v4', "Classic problem like Breakout, Pong")
flags.DEFINE_integer('screen_size', 84, 'Environment frame screen height.')
flags.DEFINE_integer("stack_history", 4, "Stack previous states.")
flags.DEFINE_integer("frame_skip", 4, "Skip n frames.")
flags.DEFINE_bool("gray_scale", True, "Gray scale observation image.")
flags.DEFINE_bool('clip_reward', True, 'Clip reward in the range [-1, 1], default on.')
flags.DEFINE_bool('done_on_life_loss', True, 'End of game if loss a life, default on.')

flags.DEFINE_integer('seed', 5, 'Seed the runtime.')

flags.DEFINE_string(
    'load_checkpoint_file',
    '',
    'Load the checkpoint from file.',
)
flags.DEFINE_string('record_video_dir', 'recordings/classic', 'Record play video.')


def main(argv):
    """Evaluates MuZero agent on Atari games."""
    del argv

    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_env = create_atari_environment(
        FLAGS.environment_name,
        FLAGS.seed,
        FLAGS.stack_history,
        FLAGS.frame_skip,
        FLAGS.screen_size,
        clip_reward=FLAGS.clip_reward,
        done_on_life_loss=FLAGS.done_on_life_loss,
        grayscale=FLAGS.gray_scale,
    )
    input_shape = eval_env.observation_space.shape
    num_actions = eval_env.action_space.n

    config = make_atari_config()

    network = MuZeroAtariNet(
        input_shape,
        num_actions,
        config.num_res_blocks,
        config.num_planes,
        config.value_support_size,
        config.reward_support_size,
    )

    # Load states from checkpoint to resume training.
    if FLAGS.load_checkpoint_file is not None and os.path.isfile(FLAGS.load_checkpoint_file):
        loaded_state = load_checkpoint(FLAGS.load_checkpoint_file, 'cpu')
        network.load_state_dict(loaded_state['network'])
        logging.info(f'Loaded state from checkpoint {FLAGS.load_checkpoint_file}')

    network.eval()

    if FLAGS.record_video_dir is not None and FLAGS.record_video_dir != '':
        full_path = f"{FLAGS.record_video_dir}_{eval_env.spec.id}"
        path = Path(full_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        eval_env = gym.wrappers.RecordVideo(eval_env, full_path)

    steps = 0
    returns = 0.0

    obs = eval_env.reset()
    while True:
        action, *_ = uct_search(
            state=obs,
            network=network,
            device=runtime_device,
            config=config,
            temperature=0.0,
            actions_mask=eval_env.actions_mask,
            current_player=eval_env.current_player,
            opponent_player=eval_env.opponent_player,
            deterministic=True,
        )

        obs, reward, done, _ = eval_env.step(action)
        steps += 1
        returns += reward

        if done:
            break

    eval_env.close()
    logging.info(f'Episode returns: {returns}, steps: {steps}')


if __name__ == '__main__':
    app.run(main)
