import pytorch_lightning as pl
import torch
from einops import rearrange, repeat

from model.feature_mlp import FeatureMLP
from model.encoder_conv import EncoderConv
from model.transformer import EncoderTransformer
from model.decoder import Decoder

from loss import pixel_likelihood_loss, latent_kl_loss


class SIMONE(pl.LightningModule):
    def __init__(
            self,
            args,
            learning_rate,
            beta_o,
            sigma_x,
            transformer_layers,
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
        target = rearrange(target, "b t c h w -> b t h w c")
        if not full_res_decode:
            # downsample targets to match model output
            target = target[:, :, ::2, ::2, :]
            target = target.index_select(dim=1, index=time_indexes)

        # Compute the individual loss terms
        pixel_likelihood_term = pixel_likelihood_loss(pixels, target, weights_softmax, self.hparams.sigma_x)
        object_latent_loss, temporal_latent_loss = latent_kl_loss(object_latents, temporal_latents)

        # Apply loss weights to get weighted loss terms
        beta_o = self.hparams.beta_o
        beta_f = 1e-4
        alpha = 0.2
        losses = {
            "pixel_likelihood_loss": alpha * pixel_likelihood_term.mean(),
            "object_latent_loss": beta_o * object_latent_loss.mean(),
            "temporal_latent_loss": beta_f * temporal_latent_loss.mean()
        }
        # Sum the weighted loss terms to get the total loss
        losses["loss"] = sum(losses.values())
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
