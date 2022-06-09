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

import torch
import torch.nn.functional as F


def signed_hyperbolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Signed hyperbolic transform, inverse of signed_parabolic."""
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def signed_parabolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Signed parabolic transform, inverse of signed_hyperbolic."""
    z = torch.sqrt(1 + 4 * eps * (eps + 1 + torch.abs(x))) / 2 / eps - 1 / 2 / eps
    return torch.sign(x) * (torch.square(z) - 1)


def normalize_hidden_state(x: torch.Tensor) -> torch.Tensor:
    """Normalize hidden state to the range [0, 1]"""
    _min = x.min(dim=1, keepdim=True)[0]
    _max = x.max(dim=1, keepdim=True)[0]
    normalized = (x - _min) / (_max - _min + 1e-8)
    return normalized


def transform_to_2hot(scalar: torch.Tensor, min_value: float, max_value: float, num_bins: int) -> torch.Tensor:
    """Transforms a scalar tensor to a 2 hot representation."""
    scalar = torch.clamp(scalar, min_value, max_value)
    scalar_bin = (scalar - min_value) / (max_value - min_value) * (num_bins - 1)
    lower, upper = torch.floor(scalar_bin), torch.ceil(scalar_bin)
    lower_value = (lower / (num_bins - 1.0)) * (max_value - min_value) + min_value
    upper_value = (upper / (num_bins - 1.0)) * (max_value - min_value) + min_value
    p_lower = (upper_value - scalar) / (upper_value - lower_value + 1e-5)
    p_upper = 1 - p_lower
    lower_one_hot = F.one_hot(lower.long(), num_bins) * torch.unsqueeze(p_lower, -1)
    upper_one_hot = F.one_hot(upper.long(), num_bins) * torch.unsqueeze(p_upper, -1)
    return lower_one_hot + upper_one_hot


def transform_from_2hot(probs: torch.Tensor, min_value: float, max_value: float, num_bins: int) -> torch.Tensor:
    """Transforms from a categorical distribution to a scalar."""
    support_space = torch.linspace(min_value, max_value, num_bins)
    support_space = support_space.expand_as(probs)
    scalar = torch.sum(probs * support_space, dim=-1, keepdim=True)
    return scalar


def logits_to_transformed_expected_value(logits: torch.Tensor, support_size: int) -> torch.Tensor:
    """
    Given raw logits (could be either reward or state value), do the following operations:
        - apply softmax
        - compute the expected scalar value
        - apply `signed_parabolic` transform function

    Args:
        logits: 2D tensor raw logits of the network output, shape [B, N].
        supports: vector of support for the computeation, shape [N,].

    Returns:
        a 2D tensor which represent the transformed expected value, shape [B, 1].
    """
    max_value = (support_size - 1) // 2
    min_value = -max_value

    # Compute expected scalar value.
    probs = torch.softmax(logits, dim=-1)
    x = transform_from_2hot(probs, min_value, max_value, support_size)

    # Apply transform funciton.
    x = signed_parabolic(x)
    return x


def scalar_to_categorical_probabilities(x: torch.Tensor, support_size: int) -> torch.Tensor:
    """
    Given scalar value (could be either reward or state value), do the following operations:
        - apply `signed_hyperbolic` transform function, which is inverse of `signed_parabolic`
        - project the values onto support base according to the MuZero paper.

    Args:
        x: 2D tensor contains the scalar values, shape [B, T].
        support_size: the full size of the support to project on to.

    Returns:
        a 3D tensor which represent the transformed and projected probabilities, shape [B, T, support_size].
    """

    # # Apply inverse transform funciton.
    x = signed_hyperbolic(x)

    max_value = (support_size - 1) // 2
    min_value = -max_value

    return transform_to_2hot(x, min_value, max_value, support_size)
