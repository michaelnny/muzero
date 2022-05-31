import collections
from typing import Callable, List, Optional

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class MuZeroConfig:
    """MuZero configuration object."""

    def __init__(
        self,
        discount: float,
        dirichlet_alpha: float,
        num_simulations: int,
        batch_size: int,
        td_steps: int,
        lr_init: float,
        lr_boundaries: List[int],
        visit_softmax_temperature_fn: Callable[[int, int], float],
        known_bounds: Optional[KnownBounds] = None,
        training_steps: Optional[int] = int(1000e3),
        num_planes: Optional[int] = 256,
        num_res_blocks: Optional[int] = 16,
        value_support_size: Optional[int] = 1,
        reward_support_size: Optional[int] = 1,
        priority_exponent: Optional[float] = 1.0,
        importance_sampling_exponent: Optional[float] = 1.0,
        train_delay: Optional[float] = 0.0,
        clip_grad: Optional[bool] = False,
        use_tensorboard: Optional[bool] = False,
        is_board_game: Optional[bool] = False,
    ) -> None:

        # Network Architecture
        self.num_planes = num_planes
        self.num_res_blocks = num_res_blocks
        self.value_support_size = value_support_size
        self.reward_support_size = reward_support_size
        self.hidden_size = 64  # for MLP network only

        # Self-Play
        self.is_board_game = is_board_game
        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_eps = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        # Training
        self.training_steps = training_steps
        self.checkpoint_interval = int(1e3)
        self.replay_capacity = 100000  # int(1e6)
        self.min_replay_size = 10000
        self.priority_exponent = priority_exponent
        self.importance_sampling_exponent = importance_sampling_exponent
        self.batch_size = batch_size
        self.unroll_steps = 5
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.adam_eps = 0.00015

        self.clip_grad = clip_grad
        self.max_grad_norm = 40.0

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_boundaries = lr_boundaries

        # Monitoring
        self.use_tensorboard = use_tensorboard

        self.train_delay = train_delay


def make_tictactoe_config(use_mlp_net: bool) -> MuZeroConfig:

    return MuZeroConfig(
        discount=1.0,
        dirichlet_alpha=0.25,
        num_simulations=30,
        batch_size=128,
        td_steps=9999,  # Always use Monte Carlo return.
        lr_init=0.002,
        lr_boundaries=[20e3],
        visit_softmax_temperature_fn=tictactoe_visit_softmax_temperature_fn,
        known_bounds=KnownBounds(-1, 1),
        training_steps=100000,
        num_planes=256 if use_mlp_net else 16,
        num_res_blocks=0 if use_mlp_net else 1,
        priority_exponent=0.0,  # Using Uniform replay
        importance_sampling_exponent=0.0,
        train_delay=0.0,
        clip_grad=False,
        use_tensorboard=True,
        is_board_game=True,
    )


def make_gomoku_config() -> MuZeroConfig:

    return MuZeroConfig(
        discount=1.0,
        dirichlet_alpha=0.03,
        num_simulations=400,
        batch_size=32,
        td_steps=9999,  # Always use Monte Carlo return.
        lr_init=0.01,
        lr_boundaries=[200e3],
        visit_softmax_temperature_fn=gomoku_visit_softmax_temperature_fn,
        known_bounds=KnownBounds(-1, 1),
        training_steps=1000000,
        num_planes=64,
        num_res_blocks=8,
        priority_exponent=0.0,  # Using Uniform replay
        importance_sampling_exponent=0.0,
        train_delay=0.0,
        clip_grad=False,
        use_tensorboard=True,
        is_board_game=True,
    )


def make_atari_config() -> MuZeroConfig:

    return MuZeroConfig(
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=30,
        batch_size=32,
        td_steps=10,
        lr_init=0.05,
        lr_boundaries=[350e3],
        visit_softmax_temperature_fn=atari_visit_softmax_temperature_fn,
        training_steps=1000000,
        num_planes=80,  # 256
        num_res_blocks=6,  # 16
        value_support_size=31,  # [-15, 15]
        reward_support_size=31,  # [-15, 15]
        train_delay=0.0,
        clip_grad=False,
        use_tensorboard=True,
        is_board_game=False,
    )


def make_classic_config() -> MuZeroConfig:

    return MuZeroConfig(
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=30,
        batch_size=128,
        td_steps=10,
        lr_init=0.005,
        lr_boundaries=[40e3],
        visit_softmax_temperature_fn=classic_visit_softmax_temperature_fn,
        training_steps=500000,
        num_planes=256,
        num_res_blocks=0,  # Always using MLP net
        value_support_size=31,  # [-15, 15]
        reward_support_size=31,  # [-15, 15]
        train_delay=0.0,
        clip_grad=False,
        use_tensorboard=True,
        is_board_game=False,
    )


def tictactoe_visit_softmax_temperature_fn(env_steps, training_steps):
    if env_steps < 6:
        return 1.0
    else:
        return 0.0  # Play according to the max.


def gomoku_visit_softmax_temperature_fn(env_steps, training_steps):
    if env_steps < 30:
        return 1.0
    else:
        return 0.0  # Play according to the max.


def atari_visit_softmax_temperature_fn(env_steps, training_steps):
    if training_steps < 500e3:
        return 1.0
    elif training_steps < 750e3:
        return 0.5
    else:
        return 0.25


def classic_visit_softmax_temperature_fn(env_steps, training_steps):
    if training_steps < 100e3:
        return 1.0
    elif training_steps < 250e3:
        return 0.5
    else:
        return 0.25
