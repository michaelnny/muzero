"""Runs MuZero self-play training pipeline on free-style Gomoku game on a small sized board.



"""
from absl import app
from absl import flags
from absl import logging
from typing import Tuple
import numpy as np
import torch

from muzero.network import MuZeroNet
from muzero.mcts import uct_search, MinMaxStats
from muzero.gym_env import create_atari_environment

FLAGS = flags.FLAGS
flags.DEFINE_string("environment_name", 'Breakout', "Classic problem like Breakout, Pong")
flags.DEFINE_integer('environment_height', 96, 'Environment frame screen height.')
flags.DEFINE_integer('environment_width', 96, 'Environment frame screen width.')
flags.DEFINE_integer("stack_history", 4, "Stack previous states.")
flags.DEFINE_bool("gray_scale", True, "Gray scale observation image.")

flags.DEFINE_integer("seed", 1, "Seed the runtime.")


def main(argv):

    env = create_atari_environment(
        FLAGS.environment_name,
        FLAGS.seed,
        FLAGS.stack_history,
        FLAGS.environment_height,
        FLAGS.environment_width,
        grayscale=FLAGS.gray_scale,
    )
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n

    runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = MuZeroNet(input_shape, num_actions).to(device=runtime_device)

    done = False
    obs = env.reset()

    steps = 0
    returns = 0.0

    while not done:
        min_max_stats = MinMaxStats()

        action, _, _ = uct_search(obs, network, runtime_device)
        obs, reward, done, info = env.step(action)
        steps += 1
        returns += reward

    env.close()

    print(f"Episode return {returns}, steps {steps}")


if __name__ == "__main__":

    app.run(main)
