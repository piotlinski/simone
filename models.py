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


class SIMONE(pl.LightningModule):
    def __init__(
        self,
        args,
        learning_rate,
        beta_o,
        sigma_x,
        transformer_layers,
        decoder_layer_norm,
        decoder_ln_scale_bias,
    ):
        super().__init__()
        self.save_hyperparameters(args)

        self.encoder_conv = EncoderConv(args)
        self.encoder_transformer = EncoderTransformer(args)

        self.feature_mlp = FeatureMLP(args)
        self.decoder = Decoder(args)

    def forward(self, x, full_res_decode=False):
        """The SIMONE model has 4 stages:
        - a convolutional encoder
        - a transformer
        - a MLP for forming the latents
        - a convolutional decoder
        """
        x = self.encoder_conv(x)
        x = self.encoder_transformer(x)
        spatial, temporal = self.feature_mlp(x)

        pixels, weights, weights_softmax, weighted_pixels, time_indexes = self.decoder(
            spatial, temporal, self.trainer, full_res_decode
        )

        return {
            "pixels": pixels,
            "weights_softmax": weights_softmax,
            "weighted_pixels": weighted_pixels,
            "object_latents": spatial,
            "temporal_latents": temporal,
            "time_indexes": time_indexes,
        }

    def loss(
        self,
        target,
        pixels,
        weights_softmax,
        object_latents,
        temporal_latents,
        time_indexes,
        full_res_decode,
        **_,
    ):
        # during validation, we decode full-size images. During training, we decode at a lower resolution
        if full_res_decode:
            xy_dim = 64
            Td = 16
        else:
            xy_dim = 32
            Td = 8

        batch_size = pixels.shape[0]

        target = rearrange(target, "b t c h w -> b t h w c", t=T, c=3, h=64, w=64)
        if not full_res_decode:
            # downsample targets to match model output
            target = target[:, :, ::2, ::2, :]
            target = target.index_select(dim=1, index=time_indexes)
            assert target.shape == (batch_size, Td, 32, 32, 3)

        assert pixels.shape == (batch_size, Td, K, xy_dim, xy_dim, 3)
        assert weights_softmax.shape == (batch_size, Td, K, xy_dim, xy_dim)

        ## Compute the pixel likelihood loss term
        sigma_x = self.hparams.sigma_x
        beta_o = self.hparams.beta_o
        beta_f = 1e-4
        alpha = 0.2
        # Expand the target in the object dimension, so it matches the shape of `pixels`
        target_expanded = target.unsqueeze(2).expand(pixels.shape)
        # Compute the log prob of each predicted pixel, for all object channels
        log_prob = torch.distributions.normal.Normal(pixels, sigma_x).log_prob(target_expanded)
        assert log_prob.shape == (batch_size, Td, K, xy_dim, xy_dim, 3)

        # Exponentiate to convert to absolute probability,
        # and take the weighted average of the pixel probabilities along the object dim using the softmax weights
        pixel_probs = torch.exp(log_prob) * weights_softmax.unsqueeze(-1)
        assert pixel_probs.shape == (batch_size, Td, K, xy_dim, xy_dim, 3)
        pixel_probs = pixel_probs.sum(dim=2)
        assert pixel_probs.shape == (batch_size, Td, xy_dim, xy_dim, 3)

        # Convert back to log space and reduce (sum) over all pixels in each batch element
        pixel_likelihood_term = (-1 / (Td * xy_dim * xy_dim)) * torch.log(pixel_probs).sum(
            dim=(4, 3, 2, 1)
        )

        ## Compute the latent loss terms
        assert pixel_likelihood_term.shape == (batch_size,)
        assert object_latents.shape == (batch_size, K, LATENT_CHANNELS, 2)
        assert temporal_latents.shape == (batch_size, T, LATENT_CHANNELS, 2), temporal_latents.shape

        # losses are the KL divergence between the predicted latent distribution
        # and the prior, which is a unit normal distribution
        object_latent_dist = get_latent_dist(object_latents)
        temporal_latent_dist = get_latent_dist(temporal_latents)
        latent_prior = torch.distributions.Normal(
            torch.zeros(object_latents.shape[:-1], device=object_latents.device, dtype=object_latents.dtype), scale=1
        )
        object_latent_term = (1 / K) * torch.distributions.kl.kl_divergence(object_latent_dist, latent_prior)
        # The KL doesn't reduce all the way because the distribution considers the batch size to be (batch, K, LATENT_CHANNELS)
        object_latent_term = object_latent_term.sum(dim=(2, 1))
        assert object_latent_term.shape == (batch_size,), object_latent_term.shape
        # this is truly T not Td
        temporal_latent_term = (1 / T) * torch.distributions.kl.kl_divergence(temporal_latent_dist, latent_prior)
        temporal_latent_term = temporal_latent_term.sum(dim=(2, 1))
        assert temporal_latent_term.shape == (batch_size,)

        # Apply loss weights to get weighted loss terms
        losses = {
            "pixel_likelihood_loss": alpha * pixel_likelihood_term.mean(),
            "object_latent_loss": beta_o * object_latent_term.mean(),
            "temporal_latent_loss": beta_f * temporal_latent_term.mean()
        }
        # Sum the weighted loss terms to get the total loss
        losses["loss"] = losses["pixel_likelihood_loss"] + losses["object_latent_loss"] + losses["temporal_latent_loss"]
        return losses

    def step(self, videos, full_res_decode=False):
        outputs = self(videos, full_res_decode)
        losses = self.loss(videos, full_res_decode=full_res_decode, **outputs)
        return {**losses, **outputs}

    def training_step(self, batch, batch_idx):
        videos, _ = batch
        outputs = self.step(videos)
        return outputs

    def validation_step(self, batch, batch_idx):
        videos, _ = batch
        outputs = self.step(videos, full_res_decode=True)
        self.log("val/loss", outputs["loss"], prog_bar=True, sync_dist=True)
        return outputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class EncoderConv(pl.LightningModule):
    """The encoder is a simple stack of convolutional layers."""
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.c1 = Conv2d(in_channels=3, out_channels=ENCODER_CONV_CHANNELS, kernel_size=4, stride=2, padding=(1, 1))
        self.c2 = Conv2d(
            in_channels=ENCODER_CONV_CHANNELS,
            out_channels=ENCODER_CONV_CHANNELS,
            kernel_size=4,
            stride=2,
            padding=(1, 1),
        )
        self.c3 = Conv2d(
            in_channels=ENCODER_CONV_CHANNELS,
            out_channels=ENCODER_CONV_CHANNELS,
            kernel_size=4,
            stride=2,
            padding=(1, 1),
        )

    def forward(self, x):
        x = rearrange(x, "b t c h w -> (b t) c h w", t=T, c=3, h=64, w=64)

        x = self.c1(x)
        x = F.relu(x)

        x = self.c2(x)
        x = F.relu(x)

        x = self.c3(x)
        x = F.relu(x)

        x = rearrange(x, "(b t) c h w -> b t c h w", t=T, c=ENCODER_CONV_CHANNELS, w=8, h=8)

        return x


class EncoderTransformer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        l = TransformerEncoderLayer(
            d_model=TRANSFORMER_CHANNELS,
            nhead=5,
            dim_feedforward=1024,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.l1 = torch.nn.Linear(in_features=ENCODER_CONV_CHANNELS, out_features=TRANSFORMER_CHANNELS, bias=False)

        self.t1 = TransformerEncoder(encoder_layer=l, num_layers=self.hparams.transformer_layers)
        self.t2 = TransformerEncoder(encoder_layer=l, num_layers=self.hparams.transformer_layers)

        self.p1 = PositionalEncoding3D(TRANSFORMER_CHANNELS)
        self.p2 = PositionalEncoding3D(TRANSFORMER_CHANNELS)

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "b t c h w -> b t h w c", t=T, h=8, w=8, c=ENCODER_CONV_CHANNELS)

        # Apply linear transformation to project ENCODER_CONV_CHANNELS to TRANSFORMER_CHANNELS
        x = self.l1(x)

        # apply 3d position encoding
        x = x + self.p1(x)

        # Convert TxWxH into a linear sequence for the transformer
        x = rearrange(x, "b t h w c -> b (t h w) c", t=T, h=8, w=8, c=TRANSFORMER_CHANNELS)

        # First transformer stage
        x = self.t1(x)

        # Do their weird pooling to reshape from TxWxH to TxK
        # Need to convert to b * t, c, h, w
        x = rearrange(x, "b (t h w) c -> (b t) c h w", t=T, h=8, w=8, c=TRANSFORMER_CHANNELS)
        x = F.avg_pool2d(x, kernel_size=2) * 4 / 2
        x = rearrange(x, "(b t) c h w -> b t h w c", t=T, h=4, w=4, c=TRANSFORMER_CHANNELS)

        # Another position encoding
        x = x + self.p2(x)

        x = rearrange(x, "b t h w c -> b (t h w) c", t=T, h=4, w=4, c=TRANSFORMER_CHANNELS)
        # Second transformer stage
        x = self.t2(x)
        assert x.shape == (batch_size, T * 4 * 4, TRANSFORMER_CHANNELS)
        return x


class FeatureMLP(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.spatial_linear_1 = torch.nn.Linear(in_features=TRANSFORMER_CHANNELS, out_features=1024)
        self.spatial_linear_2 = torch.nn.Linear(in_features=1024, out_features=LATENT_CHANNELS * 2)
        self.temporal_linear_1 = torch.nn.Linear(in_features=TRANSFORMER_CHANNELS, out_features=1024)
        self.temporal_linear_2 = torch.nn.Linear(in_features=1024, out_features=LATENT_CHANNELS * 2)

    def forward(self, x):
        # x is the output of the transformer
        x = rearrange(x, "b (t h w) c -> b t h w c", t=T, h=4, w=4, c=TRANSFORMER_CHANNELS)

        # Aggregate the temporal info to get the spatial features
        spatial = torch.mean(x, dim=1)
        spatial = rearrange(spatial, "b h w c -> (b h w) c")

        # Aggregate the spatial info to get the temporal features
        temporal = torch.mean(x, dim=(2, 3))
        temporal = rearrange(temporal, "b t c -> (b t) c", t=T, c=TRANSFORMER_CHANNELS)

        # Apply the MLPs
        spatial = self.spatial_linear_1(spatial)
        spatial = F.relu(spatial)
        spatial = self.spatial_linear_2(spatial)
        # Reshape to have 2 channels, for (mean, log_scale)
        spatial = rearrange(spatial, "(b k) (c c2) -> b k c c2", k=K, c=LATENT_CHANNELS, c2=2)

        temporal = self.temporal_linear_1(temporal)
        temporal = F.relu(temporal)
        temporal = self.temporal_linear_2(temporal)
        # Reshape to have 2 channels, for (mean, log_scale)
        temporal = rearrange(temporal, "(b t) (c c2) -> b t c c2", t=T, c=LATENT_CHANNELS, c2=2)

        return spatial, temporal


def get_latent_dist(latent, log_scale_min=-10, log_scale_max=3):
    """Convert the MLP output (with mean and log std) into a torch `Normal` distribution."""
    means = latent[..., 0]
    log_scale = latent[..., 1]
    # Clamp the minimum to keep latents from getting too far into the saturating region of the exp
    # And the max because i noticed it exploding early in the training sometimes
    log_scale = log_scale.clamp(min=log_scale_min, max=log_scale_max)
    dist = torch.distributions.normal.Normal(means, torch.exp(log_scale))
    return dist


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

    def forward(self, object_latents, temporal_latents, trainer, full_size_decode=False):
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


def create_position_encoding(range, target_shape, dim):
    """Create a tensor of shape `target_shape` that is filled with values from `range` along `dim`."""
    assert len(range.shape) == 1
    assert len(range) == target_shape[dim]

    view_shape = [1 for _ in target_shape]
    view_shape[dim] = target_shape[dim]
    range = range.view(view_shape)
    encoding = range.expand(target_shape)
    assert encoding.shape == target_shape
    return encoding
