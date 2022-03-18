import random
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from einops import repeat
from torch import Tensor

from constants import LATENT_CHANNELS
from constants import XY_RESOLUTION
from constants import K
from constants import T
from model.common import MLP
from model.common import get_latent_dist


class Decoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.mlp = MLP(in_features=LATENT_CHANNELS * 2 + 3, out_features=4, hidden_features=[512 for _ in range(5)])

    def forward(self, object_latents: Tensor, temporal_latents: Tensor, full_size_decode: bool = False):
        batch_size = object_latents.shape[0]
        if full_size_decode:
            OUTPUT_RES = XY_RESOLUTION
            Td = T
        else:
            OUTPUT_RES = XY_RESOLUTION // 2
            Td = T // 2
        assert object_latents.shape == (batch_size, K, LATENT_CHANNELS, 2)
        assert temporal_latents.shape == (batch_size, T, LATENT_CHANNELS, 2)

        # randomly downsample time axis to size Td
        # if Td=T, this will just equal `range(T)`
        time_indexes = torch.tensor(sorted(random.sample(range(T), Td)), device=self.device)

        # Downsample temporal latents along time axis
        temporal_latents = temporal_latents.index_select(dim=1, index=time_indexes)

        # for each t, k, i tuple, we want a sample from object latent k and temporal latent t.
        # these get concatted with an indicator for i and t and used as the input to the conv.
        # so the conv input is 32 + 32 + 2 + 1 channels.

        # Expand the latents to the full prediction size, so that we get a unique random sample for each pixel
        object_latents = repeat(object_latents, "b k c c2 -> b td k h w c c2", td=Td, h=OUTPUT_RES, w=OUTPUT_RES)
        temporal_latents = repeat(temporal_latents, "b td c c2 -> b td k h w c c2", k=K, h=OUTPUT_RES, w=OUTPUT_RES)

        # Draw the samples
        object_latent_samples = get_latent_dist(object_latents).rsample()
        assert object_latent_samples.shape == (batch_size, Td, K, OUTPUT_RES, OUTPUT_RES, LATENT_CHANNELS)
        temporal_latent_samples = get_latent_dist(temporal_latents).rsample()
        assert temporal_latent_samples.shape == (batch_size, Td, K, OUTPUT_RES, OUTPUT_RES, LATENT_CHANNELS)

        # Build x y t indicator features
        desired_shape = [batch_size, Td, K, OUTPUT_RES, OUTPUT_RES, 1]
        x_encoding, y_encoding, t_encoding = _build_xyt_indicators(
            desired_shape, time_indexes, self.device, object_latents.dtype, full_size_decode
        )

        # Combine all the features together
        x = torch.cat([object_latent_samples, temporal_latent_samples, t_encoding, x_encoding, y_encoding], dim=5)

        # Do the MLP
        x = rearrange(x, "b td k h w c -> (b td k h w) c", c=2 * LATENT_CHANNELS + 3)
        x = self.mlp(x)
        x = rearrange(x, "(b td k h w) c -> b td k h w c", td=Td, k=K, h=OUTPUT_RES, w=OUTPUT_RES, c=4)
        pixels = x[..., 0:3]
        weights = x[..., 3]

        # They said "apply layer norm on the decoded mask logits on the following axes simultaneously: [T, K, H, W]"
        # They also said they used a scale and bias, but it's not clear to me how that would work.
        # The usual layer norm has a scale + bias for each individual element in the tensor,
        # but since we test at higher resolution than we train, the input tensor changes shape, and this doesn't work.
        weights = torch.nn.functional.layer_norm(weights, [Td, K, OUTPUT_RES, OUTPUT_RES])

        # Apply the pixel weights
        # Softmax over the object dim and weighted average of pixels using this weight
        weights_softmax = F.softmax(weights, dim=2)
        weighted_pixels = (pixels * weights_softmax.unsqueeze(-1)).sum(dim=2)
        assert weighted_pixels.shape == (batch_size, Td, OUTPUT_RES, OUTPUT_RES, 3)

        return pixels, weights, weights_softmax, weighted_pixels, time_indexes


def _create_position_encoding(range: Tensor, target_shape: List[int], dim: int):
    """Create a tensor of shape `target_shape` that is filled with values from `range` along `dim`."""
    assert len(range.shape) == 1
    assert len(range) == target_shape[dim]

    view_shape = [1 for _ in target_shape]
    view_shape[dim] = target_shape[dim]
    range = range.view(view_shape)
    encoding = range.expand(target_shape)
    assert encoding.shape == tuple(target_shape)
    return encoding


def _build_xyt_indicators(desired_shape: List[int], time_indexes: Tensor, device, dtype, full_size_decode: bool):
    # Form the T, X, Y indicators
    t_linspace = torch.linspace(0, 1, T, device=device, dtype=dtype)
    t_linspace = t_linspace.index_select(dim=0, index=time_indexes)
    t_encoding = _create_position_encoding(t_linspace, desired_shape, dim=1)

    xy_linspace = torch.linspace(-1, 1, XY_RESOLUTION, device=device, dtype=dtype)
    if not full_size_decode:
        # we decode every other pixel
        xy_linspace = xy_linspace[::2]
    x_encoding = _create_position_encoding(xy_linspace, desired_shape, dim=3)
    y_encoding = _create_position_encoding(xy_linspace, desired_shape, dim=4)
    return x_encoding, y_encoding, t_encoding
