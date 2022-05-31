from absl import app
from absl import flags
from absl import logging
import os
import gym
import torch

from muzero.network import MuZeroBoardGameNet
from muzero.games.gomoku import GomokuEnv
from muzero.pipeline import load_checkpoint
from muzero.mcts import uct_search

FLAGS = flags.FLAGS
flags.DEFINE_integer('board_size', 9, 'Board size for Gomoku.')
flags.DEFINE_integer('num_to_win', 5, 'Number in a row to win.')
flags.DEFINE_integer('stack_history', 4, 'Stack previous states.')

flags.DEFINE_integer('num_res_blocks', 4, 'Number of res-blocks in the representation and dynamics functions.')
flags.DEFINE_integer('num_planes', 64, 'Number of planes for Conv2d layers in the model.')

flags.DEFINE_float('discount', 1.0, 'Gamma discount.')
flags.DEFINE_integer('num_simulations', 400, 'Number of simulations per MCTS search, per agent environment time step.')
flags.DEFINE_float(
    'root_noise_alpha', 0.0, 'Alph for dirichlet noise to MCTS root node prior probabilities to encourage exploration.'
)
flags.DEFINE_float('min_bound', -1.0, 'Minimum value bound.')
flags.DEFINE_float('max_bound', 1.0, 'Maximum value bound.')

flags.DEFINE_integer('seed', 1, 'Seed the runtime.')

flags.DEFINE_string(
    'load_checkpoint_file',
    'checkpoints/gomoku/Gomoku_train_steps_62000',
    'Load the checkpoint from file.',
)


def main(argv):

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    runtime_device = torch.device(device)

    eval_env = GomokuEnv(board_size=FLAGS.board_size, num_to_win=FLAGS.num_to_win, stack_history=FLAGS.stack_history)
    input_shape = eval_env.observation_space.shape
    num_actions = eval_env.action_space.n

    network = MuZeroBoardGameNet(input_shape, num_actions, FLAGS.num_res_blocks, FLAGS.num_planes)

    # Load states from checkpoint to resume training.
    if FLAGS.load_checkpoint_file is not None and os.path.isfile(FLAGS.load_checkpoint_file):
        loaded_state = load_checkpoint(FLAGS.load_checkpoint_file, 'cpu')
        network.load_state_dict(loaded_state['network'])
        logging.info(f'Loaded state from checkpoint {FLAGS.load_checkpoint_file}')

    network.eval()

    steps = 0
    returns = 0.0

    obs = eval_env.reset()
    while True:
        action, *_ = uct_search(
            state=obs,
            network=network,
            device=runtime_device,
            discount=FLAGS.discount,
            temperature=0.1,
            num_simulations=FLAGS.num_simulations,
            root_noise_alpha=FLAGS.root_noise_alpha,
            actions_mask=eval_env.actions_mask,
            current_player=eval_env.current_player,
            opponent_player=eval_env.opponent_player,
            min_bound=FLAGS.min_bound,
            max_bound=FLAGS.max_bound,
            competitive=True,
            best_action=True,
        )

        obs, reward, done, _ = eval_env.step(action)
        steps += 1
        returns += reward

        if done:
            break

    eval_env.render()
    eval_env.close()
    logging.info(f'Episode returns: {returns}, steps: {steps}')


if __name__ == '__main__':
    app.run(main)
