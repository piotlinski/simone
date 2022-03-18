import torch
from einops import repeat
from torch import Tensor

from .common import get_latent_dist


def pixel_likelihood_loss(pixels: Tensor, target: Tensor, weights_softmax: Tensor, sigma_x: float):
    """Loss based on likelihood of the pixel values, relative to a prior Normal distribution with std `sigma_x`"""
    b, t, k, h, w, c = pixels.shape
    assert target.shape == (b, t, h, w, c)
    assert weights_softmax.shape == (b, t, k, h, w)

    # Expand the target in the object dimension, so it matches the shape of `pixels`
    target = repeat(target, "b t h w c -> b t k h w c", k=k)
    # Compute the log prob of each predicted pixel, for all object channels
    log_prob = torch.distributions.normal.Normal(pixels, sigma_x).log_prob(target)
    assert log_prob.shape == (b, t, k, h, w, c)

    # Exponentiate to convert to absolute probability,
    # and take the weighted average of the pixel probabilities along the object dim using the softmax weights
    pixel_probs = torch.exp(log_prob) * weights_softmax.unsqueeze(-1)
    pixel_probs = pixel_probs.sum(dim=2)
    assert pixel_probs.shape == (b, t, h, w, c)

    # Convert back to log space and reduce (sum) over all pixels in each batch element
    pixel_likelihood_term = (-1 / (t * h * w)) * torch.log(pixel_probs).sum(dim=(4, 3, 2, 1))
    assert pixel_likelihood_term.shape == (b,)
    return pixel_likelihood_term


def latent_kl_loss(object_latents: Tensor, temporal_latents: Tensor):
    """Loss based on the KL divergence between the latents and a prior unit Normal distribution."""
    b, k, c, c2 = object_latents.shape
    b, t, c, c2 = temporal_latents.shape

    # losses are the KL divergence between the predicted latent distribution
    # and the prior, which is a unit normal distribution
    object_latent_dist = get_latent_dist(object_latents)
    temporal_latent_dist = get_latent_dist(temporal_latents)
    latent_prior = torch.distributions.Normal(
        torch.zeros(object_latents.shape[:-1], device=object_latents.device, dtype=object_latents.dtype), scale=1
    )
    object_latent_loss = (1 / k) * torch.distributions.kl.kl_divergence(object_latent_dist, latent_prior)
    # The KL doesn't reduce all the way because the distribution considers the batch size to be (batch, K, LATENT_CHANNELS)
    object_latent_loss = object_latent_loss.sum(dim=(2, 1))
    assert object_latent_loss.shape == (b,)
    temporal_latent_loss = (1 / t) * torch.distributions.kl.kl_divergence(temporal_latent_dist, latent_prior)
    temporal_latent_loss = temporal_latent_loss.sum(dim=(2, 1))
    assert temporal_latent_loss.shape == (b,)

    return object_latent_loss, temporal_latent_loss
