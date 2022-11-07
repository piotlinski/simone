import random

import pytorch_lightning as pl
import torch
import torch.distributed
from einops import rearrange
from torch import Tensor

from constants import LOG_FREQ
from contrib.deepmind.segmentation_metrics import compute_ari

from . import wandb_lib


def _generate_color_palette(num_masks: int):
    out = []
    for i in range(num_masks):
        out.append(tuple([random.randint(0, 255) for _ in range(3)]))
    return torch.tensor(out).to(torch.float) / 255


_COLORS = _generate_color_palette(16)


class WandbCallback(pl.Callback):
    """Here we just log a ton of stuff to make for easier debugging."""

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        batch, _ = batch
        spatial = outputs["object_latents"]
        temporal = outputs["temporal_latents"]
        weighted_pixels = outputs["weighted_pixels"]
        weights_softmax = outputs["weights_softmax"]
        pixels = outputs["pixels"]

        wandb_lib.log_iteration_time(trainer, self.batch_size)
        wandb_lib.log_scalar(trainer, "train/loss", outputs["loss"], freq=1)

        wandb_lib.log_video(trainer, "videos/train_dataset", batch, freq=LOG_FREQ)

        wandb_lib.log_histogram(trainer, "debug/spatial_mean", spatial[..., 0], freq=LOG_FREQ)
        wandb_lib.log_histogram(trainer, "debug/spatial_std", torch.exp(spatial[..., 1]).clamp(-1, 2), freq=LOG_FREQ)
        wandb_lib.log_histogram(trainer, "debug/spatial_log_std", spatial[..., 1], freq=LOG_FREQ)
        wandb_lib.log_histogram(trainer, "debug/temporal_mean", temporal[..., 0], freq=LOG_FREQ)
        wandb_lib.log_histogram(trainer, "debug/temporal_std", torch.exp(temporal[..., 1]).clamp(-1, 2), freq=LOG_FREQ)
        wandb_lib.log_histogram(trainer, "debug/temporal_log_std", temporal[..., 1], freq=LOG_FREQ)

        wandb_lib.log_histogram(trainer, "debug/weights_softmax", weights_softmax, freq=LOG_FREQ)
        wandb_lib.log_histogram(trainer, "debug/weighted_pixels", weighted_pixels, freq=LOG_FREQ)
        wandb_lib.log_histogram(trainer, "debug/pixels", pixels, freq=LOG_FREQ)

        # shape is (b, t, w, h, c)
        weighted_pixels_video = weighted_pixels.permute((0, 1, 4, 2, 3))
        wandb_lib.log_video(trainer, "videos/weighted_pixels", weighted_pixels_video, freq=LOG_FREQ)

        # show videos of the masks for a single batch element (so a grid of the 16 object masks)
        weights_softmax_video = weights_softmax[0].unsqueeze(-1).permute((1, 0, 4, 2, 3))
        wandb_lib.log_video(trainer, "videos/weights_softmax", weights_softmax_video, freq=LOG_FREQ)

        # show videos of the pixels for each object for a single batch element (so a grid of the 16 object masks)
        pixels_video = pixels[0].permute((1, 0, 4, 2, 3))
        wandb_lib.log_video(trainer, "videos/pixels", pixels_video, freq=LOG_FREQ)

        wandb_lib.log_scalar(trainer, "loss/pixel_likelihood", outputs["pixel_likelihood_loss"], freq=10)
        wandb_lib.log_scalar(trainer, "loss/object_latent", outputs["object_latent_loss"], freq=10)
        wandb_lib.log_scalar(trainer, "loss/temporal_latent", outputs["temporal_latent_loss"], freq=10)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        batch, _ = batch
        wandb_lib.log_video(trainer, "val/dataset", batch, freq=1)

        # loss, pixels, weights_softmax, weighted_pixels, spatial, temporal, time_indexes = outputs
        spatial = outputs["object_latents"]
        temporal = outputs["temporal_latents"]
        weighted_pixels = outputs["weighted_pixels"]
        weights_softmax = outputs["weights_softmax"]
        pixels = outputs["pixels"]

        # do all the gathering
        weighted_pixels = rearrange(pl_module.all_gather(weighted_pixels), "g b t h w c -> (g b) t h w c")
        # mask = rearrange(pl_module.all_gather(mask), "g b t k h w c -> (g b) t k h w c")
        weights_softmax = rearrange(pl_module.all_gather(weights_softmax), "g b t k h w -> (g b) t k h w")

        # shape is (b, t, w, h, c)
        weighted_pixels = weighted_pixels.permute((0, 1, 4, 2, 3))
        wandb_lib.log_video(trainer, "val/weighted_pixels", weighted_pixels[:16], freq=1)
        #
        # show videos of the masks for a single batch element (so a grid of the 16 object masks)
        # shape is (b, t, K, w, h)
        # expanded is (t, K, w, h, c)
        # Convert to (K, t, c, w, h)
        weights_softmax_video = weights_softmax[0].unsqueeze(-1).permute((1, 0, 4, 2, 3))
        wandb_lib.log_video(trainer, "val/weights_softmax", weights_softmax_video, freq=1)

        # show videos of the pixels for each object for a single batch element (so a grid of the 16 object masks)
        pixels = pixels[0].permute((1, 0, 4, 2, 3))
        wandb_lib.log_video(trainer, "val/pixels", pixels, freq=1)

        # Log segmentation mask
        segmentation = _generate_segmentation(weights_softmax, _COLORS).permute(0, 1, 4, 2, 3)
        wandb_lib.log_video(trainer, "val/segmentation", segmentation, freq=1)

        # Compute ARI
        # if torch.distributed.get_rank() == 0:
        #     ari = compute_ari(mask.cpu(), weights_softmax.cpu())
        #     wandb_lib.log_scalar(trainer, "val/ARI", ari.mean(), freq=1)

        # Log latents
        # Shape is (b, K or T, 32, 2)
        wandb_lib.log_image(trainer, "latents/obj_latent_mean", spatial[0, ..., 0].to(torch.float32), freq=1)
        wandb_lib.log_image(trainer, "latents/obj_latent_std", spatial[0, ..., 1].exp().to(torch.float32), freq=1)
        wandb_lib.log_image(trainer, "latents/temporal_latent_mean", temporal[0, ..., 0].to(torch.float32), freq=1)
        wandb_lib.log_image(
            trainer, "latents/temporal_latent_std", temporal[0, ..., 1].exp().to(torch.float32), freq=1
        )


def _generate_segmentation(weights: Tensor, colors: Tensor):
    """Generate a segmentation mask visualization."""
    colors = colors.to(weights.device)
    # weights should have shape b, t, k, h, w
    b, t, k, h, w = weights.shape
    assert len(colors) == k
    # colors should have shape k, 3
    ce = colors.view(1, 1, k, 1, 1, 3).expand(b, t, k, h, w, 3)
    wa = weights.argmax(dim=2)
    we = wa.view(b, t, 1, h, w, 1).expand(b, t, 1, h, w, 3)
    return torch.gather(ce, 2, we).view(b, t, h, w, 3)
