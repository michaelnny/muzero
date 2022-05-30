import collections
from typing import Dict, List, Optional


KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


# class MinMaxStats(object):
#     """A class that holds the min-max values of the tree."""

#     def __init__(self, known_bounds: Optional[KnownBounds]):
#         self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
#         self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

#     def update(self, value: float):
#         self.maximum = max(self.maximum, value)
#         self.minimum = min(self.minimum, value)

#     def normalize(self, value: float) -> float:
#         if self.maximum > self.minimum:
#             # We normalize only when we have set the maximum and minimum values.
#             return (value - self.minimum) / (self.maximum - self.minimum)
#         return value


class MuZeroConfig(object):
    def __init__(
        self,
        action_space_size: int,
        max_moves: int,
        discount: float,
        dirichlet_alpha: float,
        num_simulations: int,
        batch_size: int,
        td_steps: int,
        num_actors: int,
        lr_init: float,
        lr_decay_steps: float,
        visit_softmax_temperature_fn,
        known_bounds: Optional[KnownBounds] = None,
        is_board_game: bool = False,
    ):
        ### Self-Play
        self.action_space_size = action_space_size
        self.num_actors = num_actors

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        ### Training
        self.training_steps = int(1000e3)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = batch_size
        self.num_unroll_steps = 5
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps

        self.is_board_game = is_board_game


def make_gomoku_config(board_size: int) -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        if num_moves < 30:
            return 1.0
        else:
            return 0.0  # Play according to the max.

    return MuZeroConfig(
        action_space_size=board_size * 2 + 1,
        max_moves=board_size * board_size * 2,
        discount=1.0,
        dirichlet_alpha=0.03,
        num_simulations=400,
        batch_size=128,
        td_steps=board_size * board_size * 2,  # Always use Monte Carlo return.
        num_actors=3000,
        lr_init=0.01,
        lr_decay_steps=400e3,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        known_bounds=KnownBounds(-1, 1),
        is_board_game=True,
    )


def make_tictactoe_config() -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        if num_moves < 30:
            return 1.0
        else:
            return 0.0  # Play according to the max.

    return MuZeroConfig(
        action_space_size=10,
        max_moves=9,
        discount=1.0,
        dirichlet_alpha=0.03,
        num_simulations=30,
        batch_size=128,
        td_steps=9,  # Always use Monte Carlo return.
        num_actors=3000,
        lr_init=0.01,
        lr_decay_steps=10e3,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        known_bounds=KnownBounds(-1, 1),
        is_board_game=True,
    )


def make_atari_config() -> MuZeroConfig:
    def visit_softmax_temperature(num_moves, training_steps):
        if training_steps < 500e3:
            return 1.0
        elif training_steps < 750e3:
            return 0.5
        else:
            return 0.25

    return MuZeroConfig(
        action_space_size=18,
        max_moves=27000,  # Half an hour at action repeat 4.
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=50,
        batch_size=1024,
        td_steps=10,
        num_actors=350,
        lr_init=0.05,
        lr_decay_steps=350e3,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        is_board_game=False,
    )
