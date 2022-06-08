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
"""MuZero configuration components."""
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
        lr_milestones: List[int],
        visit_softmax_temperature_fn: Callable[[int, int], float],
        known_bounds: Optional[KnownBounds] = None,
        training_steps: Optional[int] = int(1000e3),
        checkpoint_interval: Optional[int] = int(1e3),
        num_planes: Optional[int] = 256,
        num_res_blocks: Optional[int] = 16,
        value_support_size: Optional[int] = 1,
        reward_support_size: Optional[int] = 1,
        priority_exponent: Optional[float] = 1.0,
        importance_sampling_exponent: Optional[float] = 1.0,
        train_delay: Optional[float] = 0.0,
        replay_capacity: Optional[int] = int(10e6),
        min_replay_size: Optional[int] = int(2e4),
        seq_length: Optional[int] = int(200),
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
        # visit_softmax_temperature_fn takes in two parameters (env_steps, train_steps)
        self.visit_softmax_temperature_fn: Callable[[int, int], float] = visit_softmax_temperature_fn
        self.num_simulations = num_simulations
        self.discount = discount
        # Send samples to learning when self-play (measured in env steps) reached this sequence length.
        self.seq_length = seq_length

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
        self.checkpoint_interval = checkpoint_interval
        # Unlike in the paper, replay capacity and min replay size are measured by single sample, not an entire game.
        self.replay_capacity = replay_capacity
        self.min_replay_size = min_replay_size

        self.priority_exponent = priority_exponent
        self.importance_sampling_exponent = importance_sampling_exponent
        self.batch_size = batch_size
        self.unroll_steps = 5
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.clip_grad = clip_grad
        self.max_grad_norm = 40.0

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_milestones = lr_milestones

        self.use_tensorboard = use_tensorboard
        self.train_delay = train_delay
        self.is_board_game = is_board_game


def make_tictactoe_config(use_mlp_net: bool = True, use_tensorboard: bool = True, clip_grad: bool = False) -> MuZeroConfig:
    """Returns MuZero config for Tic-Tac-Toe board game."""
    return MuZeroConfig(
        discount=1.0,
        dirichlet_alpha=0.25,
        num_simulations=25,
        batch_size=256,
        td_steps=0,  # Always use Monte Carlo return.
        lr_init=0.002,
        lr_milestones=[10e3],
        visit_softmax_temperature_fn=tictactoe_visit_softmax_temperature_fn,
        known_bounds=KnownBounds(-1, 1),
        training_steps=50000,
        num_planes=256 if use_mlp_net else 16,
        num_res_blocks=0 if use_mlp_net else 2,
        priority_exponent=0.0,  # Using Uniform replay
        importance_sampling_exponent=0.0,
        replay_capacity=100000,
        min_replay_size=10000,
        seq_length=9999,
        train_delay=0.0,
        clip_grad=clip_grad,
        use_tensorboard=use_tensorboard,
        is_board_game=True,
    )


def make_gomoku_config(use_tensorboard: bool = True, clip_grad: bool = False) -> MuZeroConfig:
    """Returns MuZero config for Gomoku board game."""
    return MuZeroConfig(
        discount=1.0,
        dirichlet_alpha=0.03,
        num_simulations=200,
        batch_size=64,
        td_steps=0,  # Always use Monte Carlo return.
        lr_init=0.01,
        lr_milestones=[200e3, 400e3],
        visit_softmax_temperature_fn=gomoku_visit_softmax_temperature_fn,
        known_bounds=KnownBounds(-1, 1),
        training_steps=1000000,
        num_planes=64,  # 256
        num_res_blocks=4,  # 16
        priority_exponent=0.0,  # Using Uniform replay
        importance_sampling_exponent=0.0,
        replay_capacity=100000,
        min_replay_size=20000,
        seq_length=9999,
        train_delay=0.0,
        clip_grad=clip_grad,
        use_tensorboard=use_tensorboard,
        is_board_game=True,
    )


def make_classic_config(use_tensorboard: bool = True, clip_grad: bool = False) -> MuZeroConfig:
    """Returns MuZero config for openAI Gym classic control games."""

    return MuZeroConfig(
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=30,
        batch_size=256,
        td_steps=10,
        lr_init=0.005,
        lr_milestones=[50000],
        visit_softmax_temperature_fn=classic_visit_softmax_temperature_fn,
        training_steps=200000,
        num_planes=256,
        num_res_blocks=0,  # using MLP net
        value_support_size=31,  # in the range [-15, 15]
        reward_support_size=31,  # in the range [-15, 15]
        priority_exponent=0.0,  # Using Uniform replay
        importance_sampling_exponent=0.0,
        replay_capacity=200000,
        min_replay_size=50000,
        seq_length=9999,
        train_delay=0.0,
        clip_grad=clip_grad,
        use_tensorboard=use_tensorboard,
        is_board_game=False,
    )


def make_atari_config(use_tensorboard: bool = True, clip_grad: bool = False) -> MuZeroConfig:
    """Returns MuZero config for openAI Gym image as observation version of Atari games."""
    return MuZeroConfig(
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=30,
        batch_size=8,
        td_steps=10,
        lr_init=0.005,
        lr_milestones=[100e3],
        visit_softmax_temperature_fn=atari_visit_softmax_temperature_fn,
        training_steps=1000000,
        num_planes=80,  # 256
        num_res_blocks=4,  # 16
        value_support_size=31,  # 601
        reward_support_size=31,  # 601
        priority_exponent=0.0,  # Using Uniform replay
        importance_sampling_exponent=0.0,
        replay_capacity=500000,
        min_replay_size=50000,
        seq_length=200,
        train_delay=0.0,
        clip_grad=clip_grad,
        use_tensorboard=use_tensorboard,
        is_board_game=False,
    )


def make_atari_ram_config(use_tensorboard: bool = True, clip_grad: bool = False) -> MuZeroConfig:
    """Returns MuZero config for openAI Gym RAM as observation version of Atari games."""

    return MuZeroConfig(
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=30,
        batch_size=256,
        td_steps=10,
        lr_init=0.005,
        lr_milestones=[100e3],
        visit_softmax_temperature_fn=atari_visit_softmax_temperature_fn,
        training_steps=1000000,
        num_planes=512,
        num_res_blocks=0,  # using MLP net
        value_support_size=31,
        reward_support_size=31,
        priority_exponent=0.0,  # Using Uniform replay
        importance_sampling_exponent=0.0,
        replay_capacity=500000,
        min_replay_size=50000,
        seq_length=200,
        train_delay=0.0,
        clip_grad=clip_grad,
        use_tensorboard=use_tensorboard,
        is_board_game=False,
    )


def tictactoe_visit_softmax_temperature_fn(env_steps, training_steps):
    if env_steps < 6:
        return 1.0
    # else:
    #     return 0.0  # Play according to the max.
    return 0.1


def gomoku_visit_softmax_temperature_fn(env_steps, training_steps):
    if env_steps < 30:
        return 1.0
    # else:
    #     return 0.0  # Play according to the max.
    return 0.1


def classic_visit_softmax_temperature_fn(env_steps, training_steps):
    if training_steps < 30000:
        return 1.0
    elif training_steps < 60000:
        return 0.5
    else:
        return 0.25


def atari_visit_softmax_temperature_fn(env_steps, training_steps):
    if training_steps < 250e3:
        return 1.0
    elif training_steps < 500e3:
        return 0.5
    else:
        return 0.25