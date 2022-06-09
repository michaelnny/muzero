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

from muzero.games.tictactoe import TicTacToeEnv
from muzero.network import MuZeroMLPNet, MuZeroBoardGameNet
from muzero.pipeline import load_checkpoint
from muzero.mcts import uct_search
from muzero.config import make_tictactoe_config

FLAGS = flags.FLAGS
flags.DEFINE_bool('use_mlp_net', True, 'Use FC MLP network instead Conv2d network, default on.')
flags.DEFINE_integer('seed', 5, 'Seed the runtime.')

flags.DEFINE_string(
    'load_black_checkpoint_file',
    'checkpoints/tictactoe/TicTacToe_train_steps_87000',
    'Load the last checkpoint from file.',
)
flags.DEFINE_string(
    'load_white_checkpoint_file',
    'checkpoints/tictactoe/TicTacToe_train_steps_87000',
    'Load the last checkpoint from file.',
)


def main(argv):
    """Evaluates MuZero agent on Tic-Tac-Toe game."""
    del argv

    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_env = TicTacToeEnv()
    input_shape = eval_env.observation_space.shape
    num_actions = eval_env.action_space.n

    config = make_tictactoe_config(FLAGS.use_mlp_net)

    def create_network():
        if FLAGS.use_mlp_net:
            return MuZeroMLPNet(
                input_shape,
                num_actions,
                config.num_planes,
                config.value_support_size,
                config.reward_support_size,
                config.hidden_dim,
            )
        return MuZeroBoardGameNet(input_shape, num_actions, config.num_res_blocks, config.num_planes)

    black_network = create_network().to(device=runtime_device)
    white_network = create_network().to(device=runtime_device)

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

        time.sleep(0.45)

        if done:
            break

    eval_env.close()
    logging.info(f'Episode returns: {returns}, steps: {steps}')


if __name__ == '__main__':
    app.run(main)
