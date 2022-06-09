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
import time
import torch

from muzero.network import MuZeroBoardGameNet
from muzero.games.gomoku import GomokuEnv
from muzero.config import make_gomoku_config
from muzero.pipeline import load_checkpoint
from muzero.mcts import uct_search

FLAGS = flags.FLAGS
flags.DEFINE_integer('board_size', 9, 'Board size for Gomoku.')
flags.DEFINE_integer('num_to_win', 5, 'Number in a row to win.')
flags.DEFINE_integer('stack_history', 4, 'Stack previous states.')

flags.DEFINE_integer('seed', 5, 'Seed the runtime.')

flags.DEFINE_string(
    'load_black_checkpoint_file',
    '',
    'Load the last checkpoint from file.',
)
flags.DEFINE_string(
    'load_white_checkpoint_file',
    '',
    'Load the last checkpoint from file.',
)


def main(argv):
    """Evaluates MuZero agent on Gomoku board game."""
    del argv

    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_env = GomokuEnv(board_size=FLAGS.board_size, num_to_win=FLAGS.num_to_win, stack_history=FLAGS.stack_history)
    input_shape = eval_env.observation_space.shape
    num_actions = eval_env.action_space.n

    config = make_gomoku_config()

    black_network = MuZeroBoardGameNet(input_shape, num_actions, config.num_res_blocks, config.num_planes).to(
        device=runtime_device
    )
    white_network = MuZeroBoardGameNet(input_shape, num_actions, config.num_res_blocks, config.num_planes).to(
        device=runtime_device
    )

    if FLAGS.load_black_checkpoint_file and os.path.isfile(FLAGS.load_black_checkpoint_file):
        loaded_state = load_checkpoint(FLAGS.load_black_checkpoint_file, runtime_device)
        black_network.load_state_dict(loaded_state['network'])

    if FLAGS.load_white_checkpoint_file and os.path.isfile(FLAGS.load_white_checkpoint_file):
        loaded_state = load_checkpoint(FLAGS.load_white_checkpoint_file, runtime_device)
        white_network.load_state_dict(loaded_state['network'])

    black_network.eval()
    white_network.eval()

    steps = 0
    returns = 0.0

    obs = eval_env.reset()
    while True:
        if eval_env.current_player_name == 'black':
            network = black_network
        else:
            network = white_network

        action, *_ = uct_search(
            state=obs,
            network=network,
            device=runtime_device,
            config=config,
            temperature=0.1,
            actions_mask=eval_env.actions_mask,
            current_player=eval_env.current_player,
            opponent_player=eval_env.opponent_player,
            deterministic=True,
        )

        obs, reward, done, _ = eval_env.step(action)
        eval_env.render('human')

        steps += 1
        returns += reward

        time.sleep(0.3)

        if done:
            break

    eval_env.render()
    eval_env.close()
    logging.info(f'Episode returns: {returns}, steps: {steps}')


if __name__ == '__main__':
    app.run(main)
