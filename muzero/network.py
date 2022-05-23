"""Neural Network component."""
from typing import NamedTuple, Optional, Tuple
import math
import torch
from torch import nn
import torch.nn.functional as F

from muzero.util import support_to_scalar, normalize_hidden_state


class InferenceOutputs(NamedTuple):
    """During inference, the reward and value are rescaled and transformed scalars."""

    hidden_state: torch.Tensor
    reward: torch.Tensor
    pi_probs: torch.Tensor
    value: torch.Tensor


class UnrollOutputs(NamedTuple):
    """During training, the reward and value are raw logits (no scaling or transformation)."""

    hidden_state: torch.Tensor
    reward_logits: torch.Tensor
    pi_logits: torch.Tensor
    value_logits: torch.Tensor


class MuZeroNet(nn.Module):
    """Abstract MuZero model classs."""

    def __init__(self) -> None:
        super().__init__()

    def initial_inference(self, x: torch.Tensor) -> InferenceOutputs:
        """During self-play, given environment observation, use representation function to predict initial hidden state.
        Then use prediction function to predict policy probabilities and state value (on the hidden state).
        The state value is scalar."""
        raise NotImplementedError

    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor) -> InferenceOutputs:
        """During self-play, given hidden state at timestep `t-1` and action at t,
        use dynamics function to predict the reward and next hidden state,
        and use prediction function to predict policy probabilities and state value (on new hidden state).
        The reward and state value are scalars."""
        raise NotImplementedError

    def represent(self, x: torch.Tensor) -> torch.Tensor:
        """During training, given environment observation, using representation function to predict initial hidden state."""
        raise NotImplementedError

    def prediction(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """During training, given hidden state, using prediction function to predict policy probabilities and state value.
        The state value is raw logits (no scaling or transformation)."""
        raise NotImplementedError

    def unroll_dynamics_and_prediction(self, hidden_state: torch.Tensor, action: torch.Tensor) -> UnrollOutputs:
        """During training, given hidden state and action, unroll dynamics and prediction functions,
        the output are raw logits (no scaling or transformation)."""
        raise NotImplementedError


###############################################################################
# MLP Net
###############################################################################
class RepresentationMLPNet(nn.Module):
    """Representation function with MLP NN for classic games."""

    def __init__(self, input_size: int, num_planes: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, num_planes),
            nn.ReLU(),
            nn.Linear(num_planes, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given raw state x, predict the hidden state."""
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        hidden_state = self.net(x)
        hidden_state = normalize_hidden_state(hidden_state)
        return hidden_state


class DynamicsMLPNet(nn.Module):
    """Dynamics function with MLP NN for classic games."""

    def __init__(
        self,
        num_actions: int,
        num_planes: int,
        hidden_size: int,
        support_size: int,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions

        self.state_net = nn.Sequential(
            nn.Linear(hidden_size + num_actions, num_planes),
            nn.ReLU(),
            nn.Linear(num_planes, hidden_size),
        )

        self.reward_net = nn.Sequential(
            nn.Linear(hidden_size + num_actions, num_planes),
            nn.ReLU(),
            nn.Linear(num_planes, support_size),
        )

    def forward(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given hidden_state state and action, predict the state transition and reward function."""
        assert hidden_state.shape[0] == action.shape[0]
        B = hidden_state.shape[0]

        # [batch_size, num_actions]
        onehot_action = torch.zeros((B, self.num_actions), dtype=torch.float32, device=hidden_state.device)
        onehot_action.scatter_(1, action, 1.0)

        x = torch.cat([hidden_state, onehot_action], dim=1)
        reward_logits = self.reward_net(x)

        hidden_state = self.state_net(x)
        hidden_state = normalize_hidden_state(hidden_state)
        return hidden_state, reward_logits


class PredictionMLPNet(nn.Module):
    """Prediction function with MLP NN for classic games."""

    def __init__(
        self,
        num_actions: int,
        num_planes: int,
        hidden_size: int,
        support_size: int,
    ) -> None:
        super().__init__()

        self.policy_net = nn.Sequential(
            nn.Linear(hidden_size, num_planes),
            nn.ReLU(),
            nn.Linear(num_planes, num_actions),
        )

        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, num_planes),
            nn.ReLU(),
            nn.Linear(num_planes, support_size),
        )

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given hidden state, predict the action probability distribution
        and the winning probability for current player's perspective."""

        # Predict action distributions wrt policy
        pi_logits = self.policy_net(hidden_state)

        # Predict winning probability for current player's perspective.
        value_logits = self.value_net(hidden_state)

        return pi_logits, value_logits


class MuZeroMLPNet(MuZeroNet):  # pylint: disable=abstract-method
    """MuZero model for classic games."""

    def __init__(
        self,
        input_shape: Tuple,
        num_actions: int,
        num_planes: int = 256,
        value_support_size: int = 41,
        reward_support_size: int = 11,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.value_support_size = value_support_size
        self.reward_support_size = reward_support_size

        self.represent_net = RepresentationMLPNet(math.prod(input_shape), num_planes, hidden_size)

        self.dynamics_net = DynamicsMLPNet(num_actions, num_planes, hidden_size, reward_support_size)

        self.prediction_net = PredictionMLPNet(num_actions, num_planes, hidden_size, value_support_size)

        max_value_support = (value_support_size - 1) // 2
        min_value_support = -max_value_support

        max_reward_support = (reward_support_size - 1) // 2
        min_reward_support = -max_reward_support
        self.value_supports = torch.arange(min_value_support, max_value_support + 1, dtype=torch.float32)
        self.reward_supports = torch.arange(min_reward_support, max_reward_support + 1, dtype=torch.float32)

    @torch.no_grad()
    def initial_inference(self, x: torch.Tensor) -> InferenceOutputs:
        """Given environment observation, using representation function to predict initial hidden state.
        Then using prediction function to predict policy probabilities and state value (on the hidden state)."""
        hidden_state = self.represent_net(x)

        # Prediction network
        pi_logits, value_logits = self.prediction_net(hidden_state)
        pi_probs = F.softmax(pi_logits, dim=-1)

        # Compute expected value scalar and apply rescaling.
        value = support_to_scalar(value_logits, self.value_supports.detach())
        reward = torch.zeros_like(value)
        return InferenceOutputs(hidden_state=hidden_state, reward=reward, pi_probs=pi_probs, value=value)

    @torch.no_grad()
    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor) -> InferenceOutputs:
        # Dynamics network
        hidden_state, reward_logits = self.dynamics_net(hidden_state, action)
        # Compute expected reward scalar and apply rescaling.
        reward = support_to_scalar(reward_logits, self.reward_supports.detach())

        # Prediction network
        pi_logits, value_logits = self.prediction_net(hidden_state)
        pi_probs = F.softmax(pi_logits, dim=-1)
        # Compute expected value scalar and apply rescaling.
        value = support_to_scalar(value_logits, self.value_supports.detach())
        return InferenceOutputs(hidden_state=hidden_state, reward=reward, pi_probs=pi_probs, value=value)

    def represent(self, x: torch.Tensor) -> torch.Tensor:
        return self.represent_net(x)

    def prediction(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.prediction_net(hidden_state)

    def unroll_dynamics_and_prediction(self, hidden_state: torch.Tensor, action: torch.Tensor) -> UnrollOutputs:
        # Dynamics network
        hidden_state, reward_logits = self.dynamics_net(hidden_state, action)

        # Prediction network
        pi_logits, value_logits = self.prediction_net(hidden_state)

        return UnrollOutputs(
            hidden_state=hidden_state,
            reward_logits=reward_logits,
            pi_logits=pi_logits,
            value_logits=value_logits,
        )


# ###############################################################################
# # Conv2d Net
# ###############################################################################
# class ResNetBlock(nn.Module):
#     """Basic redisual block."""

#     def __init__(
#         self,
#         num_planes: int,
#     ) -> None:
#         super().__init__()

#         self.conv_block1 = nn.Sequential(
#             nn.Conv2d(in_channels=num_planes, out_channels=num_planes, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(num_features=num_planes),
#             nn.ReLU(),
#         )

#         self.conv_block2 = nn.Sequential(
#             nn.Conv2d(in_channels=num_planes, out_channels=num_planes, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(num_features=num_planes),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         residual = x
#         out = self.conv_block1(x)
#         out = self.conv_block2(out)
#         out += residual
#         out = F.relu(out)
#         return out


# class RepresentationConvNet(nn.Module):
#     """Representation function with Conv2d NN for Atari games."""

#     def __init__(
#         self,
#         input_shape: Tuple,
#         num_planes: int,
#     ) -> None:
#         super().__init__()
#         c, h, w = input_shape

#         # output 48x48
#         self.conv_1 = nn.Conv2d(in_channels=c, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)

#         self.res_blocks_1 = nn.Sequential(*[ResNetBlock(128) for _ in range(2)])

#         # output 24x24
#         self.conv_2 = nn.Conv2d(in_channels=128, out_channels=num_planes, kernel_size=3, stride=2, padding=1, bias=False)

#         # on home computer use 2 blocks instead 3
#         self.res_blocks_2 = nn.Sequential(*[ResNetBlock(num_planes) for _ in range(2)])

#         # output 12x12
#         self.avg_pool_1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

#         # on home computer use 2 blocks instead 3
#         self.res_blocks_3 = nn.Sequential(*[ResNetBlock(num_planes) for _ in range(2)])

#         # output 6x6
#         self.avg_pool_2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Given raw state x, predict the hidden state."""
#         x = F.relu(self.conv_1(x))
#         x = self.res_blocks_1(x)
#         x = F.relu(self.conv_2(x))
#         x = self.res_blocks_2(x)
#         x = self.avg_pool_1(x)
#         x = self.res_blocks_3(x)
#         out = self.avg_pool_2(x)

#         hidden_state = normalize_hidden_state(out)
#         return hidden_state


# class DynamicsConvNet(nn.Module):
#     """Dynamics function with Conv2d NN for Atari games."""

#     def __init__(
#         self,
#         input_shape: Tuple,
#         num_actions: int,
#         num_res_block: int,
#         num_planes: int,
#         support_size: int,
#     ) -> None:
#         super().__init__()
#         self.num_actions = num_actions
#         c, h, w = input_shape

#         self.conv_block = nn.Sequential(
#             nn.Conv2d(in_channels=c, out_channels=num_planes, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(num_features=num_planes),
#             nn.ReLU(),
#         )

#         res_blocks = []
#         for _ in range(num_res_block):
#             res_block = ResNetBlock(num_planes)
#             res_blocks.append(res_block)
#         self.res_blocks = nn.Sequential(*res_blocks)

#         self.reward_head = nn.Sequential(
#             nn.Conv2d(in_channels=num_planes, out_channels=1, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(num_features=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(1 * 6 * 6, support_size),
#         )

#     def forward(self, hidden_state: torch.Tensor, action: torch.Tensor) -> DynamicsNetworkOutputs:
#         """Given hidden_state state and action, predict the state transition and reward function."""

#         assert hidden_state.shape[0] == action.shape[0]

#         b, c, h, w = hidden_state.shape

#         # [batch_size, num_actions]
#         onehot_action = F.one_hot(action.long(), self.num_actions).to(device=hidden_state.device, dtype=torch.float32)
#         # [batch_size, num_actions, h*w]
#         onehot_action = torch.repeat_interleave(onehot_action, repeats=h * w, dim=1)
#         # [batch_size, num_actions, h, w]
#         onehot_action = torch.reshape(onehot_action, (b, self.num_actions, h, w))

#         x = torch.cat([hidden_state, onehot_action], dim=1)
#         x = self.conv_block(x)
#         out = self.res_blocks(x)
#         hidden_state = normalize_hidden_state(out)

#         reward_logits = self.reward_head(hidden_state)
#         return DynamicsNetworkOutputs(hidden_state=hidden_state, reward_logits=reward_logits)


# class PredictionConvNet(nn.Module):
#     """Prediction function with Conv2d NN for Atari games."""

#     def __init__(
#         self,
#         input_shape: Tuple,
#         num_actions: int,
#         support_size: int,
#     ) -> None:
#         super().__init__()
#         c, h, w = input_shape

#         assert h == w == 6

#         half_c = c // 2

#         self.policy_head = nn.Sequential(
#             nn.Conv2d(in_channels=c, out_channels=half_c, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(num_features=half_c),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=half_c, out_channels=2, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(num_features=2),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(2 * 6 * 6, num_actions),
#         )

#         self.value_head = nn.Sequential(
#             nn.Conv2d(in_channels=c, out_channels=half_c, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(num_features=half_c),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=half_c, out_channels=1, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(num_features=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(1 * 6 * 6, support_size),
#         )

#     def forward(self, hidden_state: torch.Tensor) -> PredictionNetworkOutputs:
#         """Given hidden state, predict the action probability distribution
#         and the winning probability for current player's perspective."""

#         # Predict action distributions wrt policy
#         pi_logits = self.policy_head(hidden_state)

#         # Predict winning probability for current player's perspective.
#         value_logits = self.value_head(hidden_state)

#         return PredictionNetworkOutputs(pi_logits=pi_logits, value_logits=value_logits)


# class MuZeroConvNet(MuZeroNet):
#     """MuZero model for Atari games."""

#     def __init__(
#         self,
#         input_shape: tuple,
#         num_actions: int,
#         num_res_blocks: int = 8,
#         num_planes: int = 256,
#         support_size: int = 601,
#     ) -> None:
#         super().__init__()
#         self.num_actions = num_actions
#         self.support_size = support_size

#         support_abs_max_size = (support_size - 1) // 2

#         dynamics_input_shape = (num_planes + num_actions, 6, 6)
#         pred_input_shape = (num_planes, 6, 6)

#         self.represent_net = RepresentationConvNet(input_shape, num_planes)

#         self.dynamics_net = DynamicsConvNet(dynamics_input_shape, num_actions, num_res_blocks, num_planes, support_size)

#         self.prediction_net = PredictionConvNet(pred_input_shape, num_actions, support_size)

#         self.supports = torch.arange(-support_abs_max_size, support_abs_max_size + 1, dtype=torch.float32)

#     def initial_inference(self, x: torch.Tensor) -> InferenceOutputs:
#         """Given environment observation, using representation function to predict initial hidden state.
#         Then using prediction function to predict policy probabilities and state value (on the hidden state)."""
#         hidden_state = self.represent_net(x)

#         # Prediction network
#         prediction_out = self.prediction_net(hidden_state)
#         pi_probs = F.softmax(prediction_out.pi_logits, dim=-1)

#         # Compute expected value scalar and apply rescaling.
#         value = support_to_scalar(prediction_out.value_logits, self.supports)

#         reward = torch.zeros_like(value)

#         return InferenceOutputs(hidden_state=hidden_state, reward=reward, pi_probs=pi_probs, value=value)

#     def recurrent_inference(self, hidden_state: torch.Tensor, action: int) -> InferenceOutputs:
#         """Given hidden state at timestep t-1 and action at t, predict the reward and next hidden state at t, the policy probabilities and state value."""

#         # Prediction network
#         prediction_out = self.prediction_net(hidden_state)
#         pi_probs = F.softmax(prediction_out.pi_logits, dim=-1)

#         # Compute expected value scalar and apply rescaling.
#         value = support_to_scalar(prediction_out.value_logits, self.supports)

#         # Dynamics network
#         action = torch.tensor(action, dtype=torch.long, device=hidden_state.device)[None, ...]
#         dynamics_out = self.dynamics_net(hidden_state, action)

#         # Compute expected reward scalar and apply rescaling.
#         reward = support_to_scalar(dynamics_out.reward_logits, self.supports)

#         return InferenceOutputs(hidden_state=dynamics_out.hidden_state, reward=reward, pi_probs=pi_probs, value=value)

#     def represent(self, x: torch.Tensor) -> torch.Tensor:
#         """Given environment observation, using representation function to predict initial hidden state."""
#         return self.represent_net(x)

#     def unroll_dynamics_and_prediction(self, hidden_state: torch.Tensor, action: torch.Tensor) -> UnrollOutputs:
#         """Unroll dynamics and prediction functions during training, the output are raw logits (no scaling or transformation), not scalar."""

#         # Prediction network
#         prediction_out = self.prediction_net(hidden_state)

#         # Dynamics network
#         dynamics_out = self.dynamics_net(hidden_state, action)

#         return UnrollOutputs(
#             hidden_state=dynamics_out.hidden_state,
#             reward_logits=dynamics_out.reward_logits,
#             pi_logits=prediction_out.pi_logits,
#             value_logits=prediction_out.value_logits,
#         )
