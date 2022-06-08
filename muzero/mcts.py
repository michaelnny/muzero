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
"""MCTS class."""

from __future__ import annotations

import collections
import math
import copy
from typing import List, Tuple, Optional
import numpy as np
import torch

from muzero.network import MuZeroNet
from muzero.config import MuZeroConfig, KnownBounds


MAXIMUM_FLOAT_VALUE = float('inf')


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    """Node in the MCTS search tree."""

    def __init__(self, prior: float = None, move: int = None, parent: Node = None) -> None:
        """
        Args:
            discount: discount gamma.
            prior: a float prior probability of the node for a specific action, could be empty incase root node.
            move: the action associated to the prior probability.
            parent: the parent node, could be `None` if this is the root node.
        """
        self.player_id = None
        self.prior = prior
        self.move = move
        self.parent = parent
        self.is_expanded = False

        self.N = 0  # number of visits
        self.W = 0.0  # total action values
        self.reward = 0.0  # reward
        self.hidden_state = None  # hidden state

        self.children: List[Node] = []

    def expand(self, prior: np.ndarray, player_id: int, hidden_state: np.ndarray, reward: float) -> None:
        """Expand all actions, including illegal actions.

        Args:
            prior: 1D numpy.array contains prior probabilities of the state for all actions,
                whoever calls this should pre-processing illegal actions first.
            player_id: the current player id in the environment timestep.
            hidden_state: the corresponding hidden state for current timestep.
            reward: the reward for current timestep.

        Raises:
            ValueError:
                if node instance already expanded.
                if input argument `prior` is not a valid 1D float numpy.array.
        """
        if self.is_expanded:
            raise RuntimeError("Node already expanded.")
        if not isinstance(prior, np.ndarray) or len(prior.shape) != 1 or prior.dtype not in (np.float32, np.float64):
            raise ValueError(f"Expect `prior` to be a 1D float numpy.array, got {prior}")

        self.hidden_state = hidden_state
        self.reward = reward
        self.player_id = player_id
        for action in range(0, prior.shape[0]):
            child = Node(prior=prior[action], move=action, parent=self)
            self.children.append(child)

        self.is_expanded = True

    def best_child(self, min_max_stats: MinMaxStats, config: MuZeroConfig) -> Node:
        """Returns best child node with maximum action value Q plus an upper confidence bound U.

        Args:
            actions_mask: a 1D bool numpy.array contains legal actions for current state.
            config: a MuZeroConfig instance.

        Returns:
            The best child node.

        Raises:
            ValueError:
                if the node instance itself is a leaf node.
        """
        if not self.is_expanded:
            raise ValueError('Expand leaf node first.')

        ucb_results = self.child_Q(min_max_stats, config) + self.child_U(config)

        # Break ties when have multiple 'max' value.
        action_index = np.random.choice(np.where(ucb_results == ucb_results.max())[0])
        best_child = self.children[action_index]

        return best_child

    def backup(self, value: float, player_id: int, min_max_stats: MinMaxStats, config: MuZeroConfig) -> None:
        """Update statistics of the this node and all travesed parent nodes.

        Args:
            value: the predicted state value from NN network.
            player_id: current player in timestep t.
            min_max_stats: an MinMaxStats instance contains min and max value status regarding current MCTS run.
            is_board_game: is playing a board game.

        """

        current = self
        while current is not None:
            current.W += value if current.player_id == player_id else -value
            current.N += 1

            # What's the logic behind this??
            if current.has_parent:
                if config.is_board_game:
                    min_max_stats.update(current.reward + config.discount * -current.Q)
                else:
                    min_max_stats.update(current.reward + config.discount * current.Q)

            if config.is_board_game and current.player_id == player_id:
                value = -current.reward + config.discount * value
            else:
                value = current.reward + config.discount * value

            current = current.parent

    def child_Q(self, min_max_stats: MinMaxStats, config: MuZeroConfig) -> np.ndarray:
        """Returns a 1D numpy.array contains mean action value for all child.

        Args:
            min_max_stats: an MinMaxStats instance contains min and max value status regarding current MCTS run.
            config: a MuZeroConfig instance.

        Returns:
            a 1D numpy.array contains Q values for each of the children nodes.
        """
        p = 1.0
        if config.is_board_game:
            p = -1.0
        return np.array(
            [
                min_max_stats.normalize(child.reward + config.discount * p * child.Q) if child.N > 0 else 0
                for child in self.children
            ],
            dtype=np.float32,
        )

    def child_U(self, config: MuZeroConfig) -> np.ndarray:
        """Returns a 1D numpy.array contains UCB score for all child.

        Args:
            config: a MuZeroConfig instance.

        Returns:
            a 1D numpy.array contains UCB score for each of the children nodes.
        """
        return np.array(
            [
                child.prior
                * (
                    (math.log((self.N + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init)
                    * math.sqrt(self.N)
                    / (child.N + 1)
                )
                for child in self.children
            ],
            dtype=np.float32,
        )

    @property
    def Q(self) -> float:
        """Returns the mean action value Q(s, a)."""
        if self.N == 0:
            return 0.0
        return self.W / self.N

    @property
    def child_N(self) -> np.ndarray:
        """Returns a 1D numpy.array contains visits count for all child."""
        return np.array([child.N for child in self.children], dtype=np.int32)

    @property
    def has_parent(self) -> np.ndarray:
        """Returns a 1D numpy.array contains visits count for all child."""
        return isinstance(self.parent, Node)


def add_dirichlet_noise(prob: np.ndarray, eps: float = 0.25, alpha: float = 0.03):
    """Add dirichlet noise to a given probabilities.
    Args:
        prob: a numpy.array contains action probabilities we want to add noise to.
        eps: epsilon constant to weight the priors vs. dirichlet noise.
        alpha: parameter of the dirichlet noise distribution.

    Returns:
        action probabilities with added dirichlet noise.

    Raises:
        ValueError:
            if input argument `prob` is not a valid float numpy.array.
            if input argument `eps` or `alpha` is not float type
                or not in the range of [0.0, 1.0].
    """

    if not isinstance(prob, np.ndarray) or prob.dtype not in (np.float32, np.float64):
        raise ValueError(f"Expect `prob` to be a numpy.array, got {prob}")
    if not isinstance(eps, float) or not 0.0 <= eps <= 1.0:
        raise ValueError(f"Expect `eps` to be a float in the range [0.0, 1.0], got {eps}")
    if not isinstance(alpha, float) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Expect `alpha` to be a float in the range [0.0, 1.0], got {alpha}")

    alphas = np.ones_like(prob) * alpha
    noise = np.random.dirichlet(alphas)
    noised_prob = (1 - eps) * prob + eps * noise
    return noised_prob


def generate_play_policy(visits_count: np.ndarray, temperature: float) -> np.ndarray:
    """Returns a policy action probabilities after MCTS search,
    proportional to its exponentialted visit count.

    Args:
        visits_count: a 1D numpy.array contains child node visits count.
        temperature: a parameter controls the level of exploration.

    Returns:
        a 1D numpy.array contains the action probabilities after MCTS search.

    Raises:
        ValueError:
            if input argument `child_n` is not a valid 1D numpy.array.
            if input argument `temperature` is not float type or not in the range [0.0, 1.0].
    """
    if not isinstance(visits_count, np.ndarray) or len(visits_count.shape) != 1 or visits_count.shape == (0,):
        raise ValueError(f"Expect `visits_count` to be a 1D numpy.array, got {visits_count}")
    if not isinstance(temperature, float) or not 0.0 <= temperature <= 1.0:
        raise ValueError(f"Expect `temperature` to be float type in the range [0.0, 1.0], got {temperature}")

    visits_count = np.asarray(visits_count, dtype=np.int64)

    if temperature > 0.0:
        # We limit the exponent in the range of [1.0, 5.0]
        # to avoid overflow when doing power operation over large numbers
        exp = max(1.0, min(5.0, 1.0 / temperature))
        visits_count = np.power(visits_count, exp)

    pi_prob = visits_count / np.sum(visits_count)
    return pi_prob


def set_illegal_action_probs_to_zero(actions_mask: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Set probabilities to zero for illegal actions.
    Args:
        actions_mask: a 1D bool numpy.array with valid actions True, invalid actions False.
        prob: a 1D float numpy.array prior/action probabilities.

    Returns:
        a 1D float numpy.array prior/action probabilities with invalid actions masked out.
    """

    assert actions_mask.shape == prob.shape

    prob = np.where(actions_mask, prob, 0.0)
    sumed = np.sum(prob)
    if sumed > 0:
        prob /= sumed
    return prob


def uct_search(
    state: np.ndarray,
    network: MuZeroNet,
    device: torch.device,
    config: MuZeroConfig,
    temperature: float,
    actions_mask: np.ndarray,
    current_player: int,
    opponent_player: int,
    deterministic: bool = False,
) -> Tuple[int, np.ndarray, float]:
    """Single-threaded Upper Confidence Bound (UCB) for Trees (UCT) search without any rollout.

    It follows the following general UCT search algorithm.
    ```
    function UCTSEARCH(r,m)
      i←1
      for i ≤ m do
          n ← select(r)
          n ← expand_node(n)
          ∆ ← playout(n)
          update_statistics(n,∆)
      end for
      return end function
    ```

    Args:
        state: current observation from the environment.
        network: MuZero network instance.
        device: PyTorch runtime device.
        config: a MuZeroConfig instance.
        temperature: a parameter controls the level of exploration
            when generate policy action probabilities after MCTS search.
        actions_mask: a 1D bool numpy.array contains valid action masks, with True for valid actions, False for invalid actions.
        current_player: current player id in the environment.
        opponent_player: opponent player id in the environment.
        deterministic: after the MCTS search, choose the child node with most visits number to play in the real environment,
         instead of sample through a probability distribution, default off.

    Returns:
        tuple contains:
            a integer indicate the sampled action to play in the environment.
            a 1D numpy.array search policy action probabilities from the MCTS search result.
            a float represent the search value of the root node.

    """

    if config.is_board_game:
        assert config.discount == 1.0

    min_max_stats = MinMaxStats(config.known_bounds)

    # Create root node
    state = torch.from_numpy(state).to(device=device, dtype=torch.float32)
    network_output = network.initial_inference(state[None, ...])
    prior_prob, root_value = network_output.pi_probs, network_output.value
    root_node = Node(prior=0.0)

    # Add dirichlet noise to the prior probabilities to root node.
    if not deterministic and config.root_dirichlet_alpha > 0.0 and config.root_exploration_eps > 0.0:
        prior_prob = add_dirichlet_noise(prior_prob, eps=config.root_exploration_eps, alpha=config.root_dirichlet_alpha)
    # Set prior probabilities to zero for illegal actions.
    if actions_mask is not None:
        prior_prob = set_illegal_action_probs_to_zero(actions_mask, prior_prob)

    root_node.expand(prior_prob, current_player, network_output.hidden_state, network_output.reward)

    for _ in range(config.num_simulations):
        # Phase 1 - Select
        # Select best child node until reach a leaf node
        node = root_node
        curr_player = current_player
        oppo_player = opponent_player

        while node.is_expanded:
            node = node.best_child(min_max_stats, config)
            # Switch players as in board games.
            curr_player, oppo_player = (oppo_player, curr_player)

        # Phase 2 - Expand and evaluation
        hidden_state = torch.from_numpy(node.parent.hidden_state).to(device=device, dtype=torch.float32)
        action = torch.tensor([node.move], dtype=torch.long, device=device)
        network_output = network.recurrent_inference(hidden_state[None, ...], action[None, ...])

        node.expand(prior_prob, curr_player, network_output.hidden_state, network_output.reward)

        # Phase 3 - Backup on leaf node
        node.backup(network_output.value, curr_player, min_max_stats, config)

    # Play - generate action probability from the root node.
    child_visits = root_node.child_N
    # Maskout illegal actions.
    if actions_mask is not None:
        child_visits = np.where(actions_mask, child_visits, 0)

    pi_prob = generate_play_policy(child_visits, temperature)

    if deterministic:
        # Choose the action with most visit number.
        action_index = np.argmax(child_visits)
    else:
        # Sample a action.
        action_index = np.random.choice(np.arange(pi_prob.shape[0]), p=pi_prob)

    action = root_node.children[action_index].move
    return (action, pi_prob, root_value)
