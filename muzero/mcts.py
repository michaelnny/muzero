"""MCTS class."""

from __future__ import annotations

import math
import collections
from typing import Optional, Callable, List, Tuple, Mapping, Union, Any
import numpy as np
import torch
import torch.nn.functional as F

from muzero.network import MuZeroNet, InferenceOutputs


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

    def __init__(self, prior: float = None, move: int = None) -> None:
        """
        Args:
            player_id: the id of the current player, used to check values when we update node statistics.
            prior: a float prior probability of the node for a specific action, could be empty incase root node.
            move: the action associated to the prior probability.
            parent: the parent node, could be `None` if this is the root node.
        """

        self.prior = prior
        self.move = move
        self.is_expanded = False

        self.N = 0  # number of visits
        self.W = 0.0  # total action values
        self.reward = 0.0  # reward
        self.hidden_state = None  # hidden state

        self.player_id = None

        self.children = {}

    def value(self) -> float:
        """Returns the mean action value Q(s, a)."""
        if self.N == 0:
            return 0.0
        return self.W / self.N


def expand_node(node: Node, prior: np.ndarray, player_id: int, hidden_state: np.ndarray, reward: float) -> None:
    """Expand all actions, including illegal actions.

    Args:
        prior: 1D numpy.array contains prior probabilities of the state for all actions,
            whoever calls this should pre-processing illegal actions first.
        player_id: the current player id in the environment timestep.

    Raises:
        ValueError:
            if node instance already expanded.
            if input argument `prior` is not a valid 1D float numpy.array.
    """
    if node.is_expanded:
        raise RuntimeError("Node already expanded.")
    if not isinstance(prior, np.ndarray) or len(prior.shape) != 1 or prior.dtype not in (np.float32, np.float64):
        raise ValueError(f"Expect `prior` to be a 1D float numpy.array, got {prior}")

    node.player_id = player_id
    node.hidden_state = hidden_state
    node.reward = reward

    for action in range(0, prior.shape[0]):
        node.children[action] = Node(prior=prior[action], move=action)

    node.is_expanded = True


# Select the child with the highest UCB score.
def select_child(
    node: Node,
    min_max_stats: MinMaxStats,
    discount: float = 0.997,
    pb_c_base: float = 19652,
    pb_c_init: float = 1.25,
):
    _, action, child = max(
        (ucb_score(node, child, min_max_stats, discount, pb_c_base, pb_c_init), action, child)
        for action, child in node.children.items()
    )
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(
    parent: Node,
    child: Node,
    min_max_stats: MinMaxStats,
    discount: float,
    pb_c_base: float,
    pb_c_init: float,
) -> float:
    pb_c = math.log((parent.N + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(parent.N) / (child.N + 1)

    prior_score = pb_c * child.prior
    if child.N > 0:
        value_score = min_max_stats.normalize(child.reward + discount * child.value())
    else:
        value_score = 0
    return prior_score + value_score


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, player_id: int, discount: float, min_max_stats: MinMaxStats):
    for node in reversed(search_path):
        node.W += value if node.player_id == player_id else -value
        node.N += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


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


def generate_play_policy(child_n: np.ndarray, temperature: float) -> np.ndarray:
    """Returns a policy action probabilities after MCTS search,
    proportional to its exponentialted visit count.

    Args:
        child_n: a 1D numpy.array contains child node visits count.
        temperature: a parameter controls the level of exploration.

    Returns:
        a 1D numpy.array contains the action probabilities after MCTS search.

    Raises:
        ValueError:
            if input argument `child_n` is not a valid 1D numpy.array.
            if input argument `temperature` is not float type or not in the range (0.0, 1.0].
    """
    if not isinstance(child_n, np.ndarray) or len(child_n.shape) != 1 or child_n.shape == (0,):
        raise ValueError(f"Expect `child_n` to be a 1D numpy.array, got {child_n}")
    if not isinstance(temperature, float) or not 0.0 < temperature <= 1.0:
        raise ValueError(f"Expect `temperature` to be float type in the range (0.0, 1.0], got {temperature}")

    child_n = np.asarray(child_n, dtype=np.int64)

    # We limit the exponent in the range of [1.0, 5.0]
    # to avoid overflow when doing power operation over large numbers
    exp = max(1.0, min(5.0, 1.0 / temperature))

    pi_logits = np.power(child_n, exp)
    pi_prob = pi_logits / np.sum(pi_logits)
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
    min_bound: float = None,
    max_bound: float = None,
    best_action: bool = False,
) -> Union[int, np.ndarray, float]:
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
    if not isinstance(temperature, float) or not 0.0 < temperature <= 1.0:
        raise ValueError(f"Expect `temperature` to be float type in the range (0.0, 1.0], got {temperature}")
    if not 1 <= num_simulations:
        raise ValueError(f"Expect `num_simulations` to a positive integer, got {num_simulations}")

    min_max_stats = MinMaxStats(min_bound, max_bound)
    current_player = 1

    # Create root node
    state = torch.from_numpy(state).to(device=device, dtype=torch.float32)
    network_out = network.initial_inference(state[None, ...])
    root_value, reward, hidden_state, prior_prob = extract_network_output(network_out)

    root_node = Node()

    if root_noise_alpha is not None and root_noise_alpha > 0.0:
        prior_prob = add_dirichlet_noise(prior_prob, alpha=root_noise_alpha)

    expand_node(root_node, prior_prob, current_player, hidden_state, reward)
    # backpropagate([root_node], root_value, current_player, discount, min_max_stats)
    assert root_node.player_id == current_player

    for _ in range(num_simulations):
        # Phase 1 - Select
        # Select best child node until reach a leaf node
        node = root_node
        search_path = [node]
        while node.is_expanded:
            action, node = select_child(node, min_max_stats, discount, pb_c_base, pb_c_init)
            # history.add_action(action)
            search_path.append(node)
            # Switch players as in board games.
            # current_player, opponent_player = (opponent_player, current_player)

        # Phase 2 - Expand and evaluation
        parent = search_path[-2]
        hidden_state = torch.from_numpy(parent.hidden_state).to(device=device, dtype=torch.float32)
        action = torch.tensor(node.move, dtype=torch.long, device=device).unsqueeze(0)

        network_out = network.recurrent_inference(hidden_state[None, ...], action[None, ...])
        value, reward, hidden_state, prior_prob = extract_network_output(network_out)

        expand_node(node, prior_prob, current_player, hidden_state, reward)

        # Phase 3 - Backup on leaf node
        backpropagate(search_path, value, current_player, discount, min_max_stats)

    # Play - generate action probability from the root node.
    child_visits = np.array([child.N for child in root_node.children.values()])
    pi_prob = generate_play_policy(child_visits, temperature)

    if best_action:
        # Choose the action with most visit number.
        action = np.argmax(child_visits)
    else:
        # Sample a action.
        action = np.random.choice(np.arange(pi_prob.shape[0]), p=pi_prob)

    return (action, pi_prob, root_node.value(), abs(root_node.value() - root_value))


def extract_network_output(network_out: InferenceOutputs) -> Tuple[float, float, np.ndarray, np.ndarray]:
    pi_probs, value, reward, hidden_state = (
        network_out.pi_probs,
        network_out.value,
        network_out.reward,
        network_out.hidden_state,
    )

    # Remove batch dimensions
    pi_probs = pi_probs.squeeze(0).cpu().numpy()
    value = value.squeeze(0).cpu().item()
    reward = reward.squeeze(0).cpu().item()
    hidden_state = hidden_state.squeeze(0).cpu().numpy()

    return value, reward, hidden_state, pi_probs
