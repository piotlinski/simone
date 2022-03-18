from typing import List

import torch
from torch import Tensor
from torch import nn


class MLP(nn.Module):
    """Create a MLP with `len(hidden_features)` hidden layers, each with `hidden_features[i]` features."""

    def __init__(
        self, in_features: int, out_features: int, hidden_features: List[int], activation: nn.Module = torch.nn.ReLU()
    ):
        super().__init__()
        assert isinstance(activation, torch.nn.Module)

        layers: List[nn.Module] = []
        last_size = in_features
        for size in hidden_features:
            layers.append(nn.Linear(last_size, size))
            last_size = size
            layers.append(activation)
        # Don't put an activation after the last layer
        layers.append(nn.Linear(last_size, out_features))

        self.sequential = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.sequential(x)


def get_latent_dist(latent: Tensor, log_scale_min: float = -10, log_scale_max: float = 3):
    """Convert the MLP output (with mean and log std) into a torch `Normal` distribution."""
    means = latent[..., 0]
    log_scale = latent[..., 1]
    # Clamp the minimum to keep latents from getting too far into the saturating region of the exp
    # And the max because I noticed it exploding early in the training sometimes
    log_scale = log_scale.clamp(min=log_scale_min, max=log_scale_max)
    dist = torch.distributions.normal.Normal(means, torch.exp(log_scale))
    return dist
