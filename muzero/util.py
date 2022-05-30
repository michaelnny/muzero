import torch


def signed_hyperbolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Signed hyperbolic transform, inverse of signed_parabolic."""
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def signed_parabolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Signed parabolic transform, inverse of signed_hyperbolic."""
    z = torch.sqrt(1 + 4 * eps * (eps + 1 + torch.abs(x))) / 2 / eps - 1 / 2 / eps
    return torch.sign(x) * (torch.square(z) - 1)


def normalize_hidden_state(x: torch.Tensor) -> torch.Tensor:
    """Normalize hidden state to the range [0, 1]"""
    orig_shape = x.shape

    x_flatten = torch.flatten(x, 1, -1)

    _min = torch.min(x_flatten, dim=1, keepdim=True)[0]
    _max = torch.max(x_flatten, dim=1, keepdim=True)[0]
    normalized = (x_flatten - _min) / (_max - _min)
    normalized = normalized.view(orig_shape)
    return normalized


def logits_to_transformed_expected_value(logits: torch.Tensor, supports: torch.Tensor) -> torch.Tensor:
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

    # Compute expected scalar.
    probs = torch.softmax(logits, dim=1)
    support = supports.to(device=logits.device, dtype=probs.dtype).expand(probs.shape)
    x = torch.sum(support.detach() * probs, dim=1, keepdim=True)

    # Apply transform funciton.
    x = signed_parabolic(x)
    return x


def scalar_to_categorical_probabilities(x: torch.Tensor, support_size: int) -> torch.Tensor:
    """
    Given scalar value (could be either reward or state value), do the following operations:
        - apply inverse of `signed_hyperbolic`
        - project the values onto support base according to the MuZero paper.

    Args:
        x: 2D tensor contains the scalar values, shape [B, T].
        support_size: the size of the support to project on to.

    Returns:
        a 3D tensor which represent the transformed and projected probabilities, shape [B, T, support_size].
    """

    # Apply inverse transform funciton.
    x = signed_hyperbolic(x)

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
