import pytorch_lightning as pl
import torch
from einops import rearrange, repeat

from config import ENCODER_CONV_CHANNELS, TRANSFORMER_CHANNELS, LATENT_CHANNELS, K, T

from model.feature_mlp import FeatureMLP
from model.encoder_conv import EncoderConv
from model.transformer import EncoderTransformer
from model.decoder import Decoder
from util import get_latent_dist
from loss import pixel_likelihood_loss


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
            spatial, temporal, full_res_decode
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
