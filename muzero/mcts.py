"""MCTS class."""

from __future__ import annotations

import math
import collections
import copy
from typing import Optional, Callable, List, Tuple, Mapping, Union, Any
import numpy as np
import torch
import torch.nn.functional as F

from muzero.network import MuZeroNet, NetworkOutputs


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, minimum_bound=None, maximum_bound=None):
        self.minimum = float('inf') if minimum_bound is None else minimum_bound
        self.maximum = -float('inf') if maximum_bound is None else maximum_bound

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

    def __init__(self, player_id: int, discount: float, prior: float = None, move: int = None, parent: Node = None) -> None:
        """
        Args:
            player_id: the id of the current player, used to check values when we update node statistics.
            discount: discount gamma.
            prior: a float prior probability of the node for a specific action, could be empty incase root node.
            move: the action associated to the prior probability.
            parent: the parent node, could be `None` if this is the root node.
        """
        self.player_id = player_id
        self.prior = prior
        self.move = move
        self.parent = parent
        self.is_expanded = False

        self.N = 0  # number of visits
        self.W = 0.0  # total action values
        self.reward = 0.0  # reward
        self.hidden_state = None  # hidden state

        self.discount = discount

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

        for action in range(0, prior.shape[0]):
            child = Node(player_id=player_id, discount=self.discount, prior=prior[action], move=action, parent=self)
            self.children.append(child)

        self.is_expanded = True

    def best_child(
        self,
        min_max_stats: MinMaxStats,
        pb_c_base: float = 19652,
        pb_c_init: float = 1.25,
        is_board_game: bool = False,
    ) -> Node:
        """Returns best child node with maximum action value Q plus an upper confidence bound U.

        Args:
            actions_mask: a 1D bool numpy.array contains legal actions for current state.
            c_puct: a float constatnt determining the level of exploration, default 5.0.

        Returns:
            The best child node.

        Raises:
            ValueError:
                if the node instance itself is a leaf node.
                if input argument `actions_mask` is not a valid 1D bool numpy.array.
                if input argument `c_puct` is not float type or not in the range [0.0, 10.0].
        """
        if not self.is_expanded:
            raise ValueError('Expand leaf node first.')

        # if is_board_game:
        #     ucb_results = -1.0 * self.child_Q_old() + 5.0 * self.child_U_old()
        #     # ucb_results = -1.0 * self.child_Q(min_max_stats, is_board_game) + self.child_U(pb_c_base, pb_c_init)
        # else:
        ucb_results = self.child_Q(min_max_stats, is_board_game) + self.child_U(pb_c_base, pb_c_init)

        # Break ties when have multiple 'max' value.
        best_action = np.random.choice(np.where(ucb_results == ucb_results.max())[0])
        best_child = self.children[best_action]

        return best_child

    def backup(self, value: float, player_id: int, min_max_stats: MinMaxStats, is_board_game: bool) -> None:
        """Update statistics of the this node and all travesed parent nodes."""

        current = self
        while current is not None:
            current.W += value if current.player_id == player_id else -value
            current.N += 1

            if is_board_game:
                # q = current.reward - current.discount * current.Q
                pass
            else:
                min_max_stats.update(current.reward + current.discount * current.Q)
                value = current.reward + current.discount * value

            # if is_board_game and current.player_id == player_id:
            #     reward = -current.reward
            # else:
            #     reward = current.reward

            current = current.parent

    def child_Q(self, min_max_stats: MinMaxStats, is_board_game: bool) -> np.ndarray:
        """Returns a 1D numpy.array contains mean action value for all child."""
        p = 1.0
        if is_board_game:
            p = -1.0
            return -1.0 * np.array([child.Q for child in self.children], dtype=np.float32)
        return np.array(
            [min_max_stats.normalize(child.reward + child.discount * child.Q) for child in self.children], dtype=np.float32
        )

    def child_U(self, pb_c_base: float, pb_c_init: float) -> np.ndarray:
        """Returns a 1D numpy.array contains UCB score for all child."""
        # return np.array([child.prior * (math.sqrt(self.N) / (1 + child.N)) for child in self.children], dtype=np.float32)

        return np.array(
            [
                child.prior
                * ((math.log((self.N + pb_c_base + 1) / pb_c_base) + pb_c_init) * math.sqrt(self.N) / (child.N + 1))
                for child in self.children
            ],
            dtype=np.float32,
        )

    # def child_Q_old(self) -> np.ndarray:
    #     """Returns a 1D numpy.array contains UCB score for all child."""
    #     return np.array([child.Q for child in self.children], dtype=np.float32)

    # def child_U_old(self) -> np.ndarray:
    #     """Returns a 1D numpy.array contains UCB score for all child."""
    #     return np.array([child.prior * (math.sqrt(self.N) / (1 + child.N)) for child in self.children], dtype=np.float32)

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

        pi_logits = np.power(visits_count, exp)
        pi_prob = pi_logits / np.sum(pi_logits)
    else:
        pi_prob = visits_count / np.sum(visits_count)

    return pi_prob


def set_illegal_action_probs_to_zero(actions_mask: np.ndarray, prob: np.ndarray) -> np.ndarray:
    # Set probabilities to zero for illegal actions.
    prob = np.where(actions_mask, prob, 0.0)
    sumed = np.sum(prob)
    if sumed > 0:
        prob /= sumed
    return prob


@torch.no_grad()
def uct_search(
    state: np.ndarray,
    network: MuZeroNet,
    device: torch.device,
    discount: float = 0.997,
    pb_c_base: float = 19652,
    pb_c_init: float = 1.25,
    temperature: float = 1.0,
    num_simulations: int = 50,
    root_noise_alpha: float = None,
    actions_mask: np.ndarray = None,
    current_player: int = 1,
    opponent_player: int = 1,
    is_board_game: bool = False,
    min_bound: float = None,
    max_bound: float = None,
    best_action: bool = False,
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
    """
    if not isinstance(temperature, float) or not 0.0 <= temperature <= 1.0:
        raise ValueError(f"Expect `temperature` to be float type in the range [0.0, 1.0], got {temperature}")
    if num_simulations < 1:
        raise ValueError(f"Expect `num_simulations` to a positive integer, got {num_simulations}")

    init_current_player = copy.deepcopy(current_player)
    init_opponent_player = copy.deepcopy(opponent_player)
    actions_mask = copy.deepcopy(actions_mask)

    min_max_stats = MinMaxStats(min_bound, max_bound)

    # Create root node
    state = torch.from_numpy(state).to(device=device, dtype=torch.float32)
    network_out = network.initial_inference(state[None, ...])
    root_value, reward, hidden_state, prior_prob = extract_network_output(network_out, is_board_game)

    root_node = Node(player_id=init_current_player, discount=discount)

    # Add dirichlet noise to the prior probabilities to root node.
    if root_noise_alpha is not None and root_noise_alpha > 0.0:
        prior_prob = add_dirichlet_noise(prior_prob, alpha=root_noise_alpha)
    # Set prior probabilities to zero for illegal actions.
    if actions_mask is not None:
        prior_prob = set_illegal_action_probs_to_zero(actions_mask, prior_prob)

    root_node.expand(prior_prob, init_opponent_player, hidden_state, reward)
    root_node.backup(root_value, init_current_player, min_max_stats, is_board_game)
    assert root_node.player_id == current_player

    for _ in range(num_simulations):
        # Phase 1 - Select
        # Select best child node until reach a leaf node
        node = root_node
        curr_player = init_current_player
        oppo_player = init_opponent_player

        while node.is_expanded:
            node = node.best_child(min_max_stats, pb_c_base, pb_c_init, is_board_game)
            # Switch players as in board games.
            curr_player, oppo_player = (oppo_player, curr_player)

        # Phase 2 - Expand and evaluation
        hidden_state = torch.from_numpy(node.parent.hidden_state).to(device=device, dtype=torch.float32)
        action = torch.tensor(node.move, dtype=torch.long, device=device).unsqueeze(0)

        network_out = network.recurrent_inference(hidden_state[None, ...], action[None, ...])
        value, reward, hidden_state, prior_prob = extract_network_output(network_out, is_board_game)

        node.expand(prior_prob, oppo_player, hidden_state, reward)

        # Phase 3 - Backup on leaf node
        node.backup(value, curr_player, min_max_stats, is_board_game)

    # Play - generate action probability from the root node.
    child_visits = root_node.child_N
    # Maskout illegal actions.
    if actions_mask is not None:
        child_visits = np.where(actions_mask, child_visits, 0)

    pi_prob = generate_play_policy(child_visits, temperature)

    if best_action:
        # Choose the action with most visit number.
        action = np.argmax(child_visits)
    else:
        # Sample a action.
        action = np.random.choice(np.arange(pi_prob.shape[0]), p=pi_prob)

    return (action, pi_prob, root_node.Q, abs(root_node.Q - root_value))


def extract_network_output(network_out: NetworkOutputs, is_board_game: bool) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Extract network inference output."""
    pi_probs = F.softmax(network_out.pi_logits, dim=1)

    # Remove batch dimensions and turn into numpy or scalar values.
    pi_probs = pi_probs.squeeze(0).cpu().numpy()
    value = network_out.value.squeeze(0).cpu().item()
    # Board game has no immediate reward.
    reward = 0.0 if is_board_game else network_out.reward.squeeze(0).cpu().item()
    # reward = network_out.reward.squeeze(0).cpu().item()
    hidden_state = network_out.hidden_state.squeeze(0).cpu().numpy()

    return value, reward, hidden_state, pi_probs
