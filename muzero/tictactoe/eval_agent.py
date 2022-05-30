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

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_res_blocks', 2, 'Number of res-blocks in the representation and dynamics functions.')
flags.DEFINE_integer('num_planes', 512, 'Number of planes for Conv2d layers in the model.')

flags.DEFINE_float('discount', 1.0, 'Gamma discount.')
flags.DEFINE_integer('num_simulations', 30, 'Number of simulations per MCTS search, per agent environment time step.')
flags.DEFINE_float(
    'root_noise_alpha', 0.0, 'Alph for dirichlet noise to MCTS root node prior probabilities to encourage exploration.'
)
flags.DEFINE_float('pb_c_base', 19652.0, 'PB C Base.')
flags.DEFINE_float('pb_c_init', 1.25, 'PB C Init.')
flags.DEFINE_float('min_bound', -1.0, 'Minimum value bound.')
flags.DEFINE_float('max_bound', 1.0, 'Maximum value bound.')

flags.DEFINE_integer('seed', 1, 'Seed the runtime.')

flags.DEFINE_string(
    'load_black_checkpoint_file',
    'checkpoints/tictactoe/TicTacToe_train_steps_38000',
    'Load the last checkpoint from file.',
)
flags.DEFINE_string(
    'load_white_checkpoint_file',
    'checkpoints/tictactoe/TicTacToe_train_steps_38000',
    'Load the last checkpoint from file.',
)


def main(argv):

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    runtime_device = torch.device(device)

    eval_env = TicTacToeEnv()
    input_shape = eval_env.observation_space.shape
    num_actions = eval_env.action_space.n

    # black_network = MuZeroBoardGameNet(input_shape, num_actions, FLAGS.num_res_blocks, FLAGS.num_planes).to(
    #     device=runtime_device
    # )
    # white_network = MuZeroBoardGameNet(input_shape, num_actions, FLAGS.num_res_blocks, FLAGS.num_planes).to(
    #     device=runtime_device
    # )
    black_network = MuZeroMLPNet(input_shape, num_actions, FLAGS.num_planes, 31, 31, 64).to(device=runtime_device)
    white_network = MuZeroMLPNet(input_shape, num_actions, FLAGS.num_planes, 31, 31, 64).to(device=runtime_device)

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
            discount=FLAGS.discount,
            pb_c_base=FLAGS.pb_c_base,
            pb_c_init=FLAGS.pb_c_init,
            temperature=0.1,
            num_simulations=FLAGS.num_simulations,
            root_noise_alpha=FLAGS.root_noise_alpha,
            actions_mask=eval_env.actions_mask,
            current_player=eval_env.current_player,
            opponent_player=eval_env.opponent_player,
            min_bound=FLAGS.min_bound,
            max_bound=FLAGS.max_bound,
            is_board_game=True,
            best_action=True,
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
