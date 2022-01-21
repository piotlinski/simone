import random

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import Conv2d
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
from einops import rearrange, repeat

from util import PositionalEncoding3D
from config import ENCODER_CONV_CHANNELS, TRANSFORMER_CHANNELS, LATENT_CHANNELS, K, T


class Decoder(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        # All convolutions have stride 1 and kernel size 1,
        # as this is really just a MLP over the channel dimension
        self.c1 = Conv2d(in_channels=LATENT_CHANNELS * 2 + 3, out_channels=512, kernel_size=1, stride=1)
        self.convs = torch.nn.ModuleList(
            [Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1) for _ in range(4)]
        )
        self.c6 = Conv2d(in_channels=512, out_channels=4, kernel_size=1, stride=1)

        # Parameters for the non-standard layer norm scale + bias
        if self.hparams.decoder_ln_scale_bias:
            self.ln_weight = torch.nn.Parameter(torch.tensor(1.0))
            self.ln_bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, object_latents, temporal_latents, full_size_decode=False):
        if full_size_decode:
            OUTPUT_RES = 64
            Td = 16
        else:
            OUTPUT_RES = 32
            Td = 8

        # randomly downsample time axis to size Td
        if not full_size_decode:
            time_indexes = torch.tensor(sorted(random.sample(range(T), Td)), device=self.device)
        else:
            time_indexes = None

        batch_size = object_latents.shape[0]
        assert object_latents.shape == (batch_size, K, LATENT_CHANNELS, 2)
        assert temporal_latents.shape == (batch_size, T, LATENT_CHANNELS, 2)
        if not full_size_decode:
            # Downsample temporal latents along time axis also
            temporal_latents = temporal_latents.index_select(dim=1, index=time_indexes)

        # for each t, k, i tuple, we want a sample from object latent k and temporal latent t.
        # these get concatted with an indicator for i and t and used as the input to the conv.
        # so the conv input is 32 + 32 + 2 + 1 channels.

        # Expand the latents to the full prediction size, so that we get a unique random sample for each pixel
        object_latents = repeat(object_latents, "b k c c2 -> b td k h w c c2", k=K, td=Td, h=OUTPUT_RES, w=OUTPUT_RES, c=LATENT_CHANNELS, c2=2)
        temporal_latents = repeat(temporal_latents, "b td c c2 -> b td k h w c c2", k=K, td=Td, h=OUTPUT_RES, w=OUTPUT_RES, c=LATENT_CHANNELS, c2=2)

        # Draw the samples
        object_latent_samples = get_latent_dist(object_latents).rsample()
        assert object_latent_samples.shape == (batch_size, Td, K, OUTPUT_RES, OUTPUT_RES, LATENT_CHANNELS)
        temporal_latent_samples = get_latent_dist(temporal_latents).rsample()
        assert temporal_latent_samples.shape == (batch_size, Td, K, OUTPUT_RES, OUTPUT_RES, LATENT_CHANNELS)

        # Form the T, X, Y indicators
        desired_shape = (batch_size, Td, K, OUTPUT_RES, OUTPUT_RES, 1)
        t_linspace = torch.linspace(0, 1, T, device=self.device, dtype=object_latents.dtype)
        if not full_size_decode:
            t_linspace = t_linspace.index_select(dim=0, index=time_indexes)
        t_encoding = create_position_encoding(t_linspace, desired_shape, dim=1)

        xy_linspace = torch.linspace(-1, 1, 64, device=self.device, dtype=object_latents.dtype)
        if not full_size_decode:
            # we decode every other pixel
            xy_linspace = xy_linspace[::2]
        x_encoding = create_position_encoding(xy_linspace, desired_shape, dim=3)
        y_encoding = create_position_encoding(xy_linspace, desired_shape, dim=4)

        # Combine all the features together
        x = torch.cat([object_latent_samples, temporal_latent_samples, t_encoding, x_encoding, y_encoding], dim=5)

        assert x.shape == (batch_size, Td, K, OUTPUT_RES, OUTPUT_RES, 2 * LATENT_CHANNELS + 3), x.shape

        # Do the convolution stack
        x = rearrange(x, "b td k h w c -> (b td k) c h w", c=2 * LATENT_CHANNELS + 3)
        x = self.c1(x)
        x = F.relu(x)

        for conv in self.convs:
            x = conv(x)
            x = F.relu(x)

        x = self.c6(x)
        assert x.shape == (batch_size * Td * K, 4, OUTPUT_RES, OUTPUT_RES)

        # Convert back to preferred shape
        x = rearrange(x, "(b td k) c h w -> b td k h w c", td=Td, k=K, c=4)
        pixels = x[..., 0:3]

        # They said "apply layer norm on the decoded mask logits on the following axes simultaneously: [T, K, H, W]"
        # They also said they used a scale and bias, but it's not clear to me how that would work.
        # The usual layer norm has a scale + bias for each individual element in the tensor,
        # but since we test at higher resolution than we train, the input tensor changes shape, and this doesn't work.
        # I've added a feature flag to apply a single mean + scale to the entire tensor,
        # but have mainly left this disabled and still get good results.
        weights = x[..., 3]
        if self.hparams.decoder_layer_norm:
            if self.hparams.decoder_ln_scale_bias:
                weight = self.ln_weight.reshape(1, 1, 1, 1).expand(Td, K, OUTPUT_RES, OUTPUT_RES)
                bias = self.ln_bias.reshape(1, 1, 1, 1).expand(Td, K, OUTPUT_RES, OUTPUT_RES)
                # wandb_lib.log_scalar(trainer, "debug/ln_weight", self.ln_weight.data, freq=LOG_FREQ)
                # wandb_lib.log_scalar(trainer, "debug/ln_bias", self.ln_bias.data, freq=LOG_FREQ)
                weights = F.layer_norm(weights, [Td, K, OUTPUT_RES, OUTPUT_RES], weight, bias)
            else:
                weights = torch.nn.functional.layer_norm(weights, [Td, K, OUTPUT_RES, OUTPUT_RES])

        assert pixels.shape == (batch_size, Td, K, OUTPUT_RES, OUTPUT_RES, 3)
        assert weights.shape == (batch_size, Td, K, OUTPUT_RES, OUTPUT_RES)

        # Apply the pixel weights
        # Softmax over the object dim and weighted average of pixels using this weight
        weights_softmax = F.softmax(weights, dim=2)
        weighted_pixels = (pixels * weights_softmax.unsqueeze(-1)).sum(dim=2)
        assert weighted_pixels.shape == (batch_size, Td, OUTPUT_RES, OUTPUT_RES, 3)

        return pixels, weights, weights_softmax, weighted_pixels, time_indexes
