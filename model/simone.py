import pytorch_lightning as pl
import torch
from einops import rearrange
from torch import Tensor

from model.decoder import Decoder
from model.encoder_conv import EncoderConv
from model.feature_mlp import FeatureMLP
from model.loss import latent_kl_loss
from model.loss import pixel_likelihood_loss
from model.transformer import EncoderTransformer


class SIMONE(pl.LightningModule):
    def __init__(
        self,
        learning_rate,
        transformer_layers,
        sigma_x,
        alpha,
        beta_o,
        beta_f,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        # This saves the input args into self.hparams
        self.save_hyperparameters()

        self.encoder_conv = EncoderConv()
        self.encoder_transformer = EncoderTransformer(transformer_layers)
        self.feature_mlp = FeatureMLP()
        self.decoder = Decoder()

    def forward(self, input_videos: Tensor, full_res_decode: bool = False):
        """The SIMONE model has 4 stages:
        - a convolutional encoder
        - a transformer
        - a MLP for forming the latents
        - a convolutional decoder
        """
        conv_output = self.encoder_conv(input_videos)
        transformer_output = self.encoder_transformer(conv_output)
        spatial, temporal = self.feature_mlp(transformer_output)

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
        target: Tensor,
        pixels: Tensor,
        weights_softmax: Tensor,
        object_latents: Tensor,
        temporal_latents: Tensor,
        time_indexes: Tensor,
        full_res_decode: bool,
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
        losses = {
            "pixel_likelihood_loss": self.hparams.alpha * pixel_likelihood_term.mean(),
            "object_latent_loss": self.hparams.beta_o * object_latent_loss.mean(),
            "temporal_latent_loss": self.hparams.beta_f * temporal_latent_loss.mean(),
        }
        # Sum the weighted loss terms to get the total loss
        losses["loss"] = sum(losses.values())
        return losses

    def step(self, videos: Tensor, full_res_decode: bool = False):
        outputs = self(videos, full_res_decode)
        losses = self.loss(videos, full_res_decode=full_res_decode, **outputs)
        return {**losses, **outputs}

    def training_step(self, batch: Tensor, batch_idx: int):
        videos, _ = batch
        outputs = self.step(videos)
        return outputs

    def validation_step(self, batch: Tensor, batch_idx: int):
        videos, _ = batch
        outputs = self.step(videos, full_res_decode=True)
        self.log("val/loss", outputs["loss"], prog_bar=True, sync_dist=True)
        return outputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
