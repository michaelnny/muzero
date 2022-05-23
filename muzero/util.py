from typing import NamedTuple, Optional, Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def signed_hyperbolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Signed hyperbolic transform, inverse of signed_parabolic."""
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def signed_parabolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Signed parabolic transform, inverse of signed_hyperbolic."""
    z = torch.sqrt(1 + 4 * eps * (eps + 1 + torch.abs(x))) / 2 / eps - 1 / 2 / eps
    return torch.sign(x) * (torch.square(z) - 1)


def normalize_hidden_state(x: torch.Tensor) -> torch.Tensor:
    """Normalize hidden state (by channel) in range [0, 1]"""
    _min = torch.min(x, dim=1, keepdim=True)[0]
    _max = torch.max(x, dim=1, keepdim=True)[0]
    return (x - _min) / (_max - _min + 1e-8)


def support_to_scalar(logits: torch.Tensor, supports: torch.Tensor) -> torch.Tensor:
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """

    # Compute expected scalar.
    probs = torch.softmax(logits, dim=1)
    support = supports.to(device=logits.device, dtype=probs.dtype).expand(probs.shape)
    x = torch.sum(support.detach() * probs, dim=1, keepdim=True)

    # Apply rescaling funciton.
    x = signed_parabolic(x)
    return x


def scalar_to_support(x: torch.Tensor, support_size: int) -> torch.Tensor:
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """

    # Apply inverse rescaling funciton.
    x = signed_hyperbolic(x)

    # B, T = x.shape
    max_size = (support_size - 1) // 2
    min_size = -max_size

    x.clamp_(min_size, max_size)
    x_low = x.floor()
    x_high = x.ceil()
    p_high = x - x_low
    p_low = 1 - p_high

    support = torch.zeros(x.shape[0], x.shape[1], support_size).to(x.device)
    x_high_idx, x_low_idx = x_high - min_size, x_low - min_size
    support.scatter_(2, x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
    support.scatter_(2, x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))

    return support
