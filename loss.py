import torch
from einops import repeat
from util import check_shape

from einops import parse_shape
from util import get_latent_dist


def pixel_likelihood_loss(pixels, target, weights_softmax, sigma_x):
    """Loss based on likelihood of the pixel values, relative to a prior Normal distribution with std `sigma_x`"""
    shapes = parse_shape(pixels, "b t k h w c")
    check_shape(target, "b t h w c", **shapes)
    check_shape(weights_softmax, "b t k h w", **shapes)

    # Expand the target in the object dimension, so it matches the shape of `pixels`
    target = repeat(target, "b t h w c -> b t k h w c", **shapes)
    # Compute the log prob of each predicted pixel, for all object channels
    log_prob = torch.distributions.normal.Normal(pixels, sigma_x).log_prob(target)
    check_shape(log_prob, "b t k h w c", **shapes)

    # Exponentiate to convert to absolute probability,
    # and take the weighted average of the pixel probabilities along the object dim using the softmax weights
    pixel_probs = torch.exp(log_prob) * weights_softmax.unsqueeze(-1)
    pixel_probs = pixel_probs.sum(dim=2)
    check_shape(pixel_probs, "b t h w c", **shapes)

    # Convert back to log space and reduce (sum) over all pixels in each batch element
    pixel_likelihood_term = (-1 / (shapes["t"] * shapes["h"] * shapes["w"])) * torch.log(pixel_probs).sum(dim=(4, 3, 2, 1))
    check_shape(pixel_likelihood_term, "b", **shapes)

    return pixel_likelihood_term


def latent_kl_loss(object_latents, temporal_latents):
    """Loss based on the KL divergence between the latents and a prior unit Normal distribution."""
    shapes = parse_shape(object_latents, "b k c c2") | parse_shape(temporal_latents, "b t c c2")
    check_shape(object_latents, "b k c c2", **shapes)
    check_shape(temporal_latents, "b t c c2", **shapes)

    # losses are the KL divergence between the predicted latent distribution
    # and the prior, which is a unit normal distribution
    object_latent_dist = get_latent_dist(object_latents)
    temporal_latent_dist = get_latent_dist(temporal_latents)
    latent_prior = torch.distributions.Normal(
        torch.zeros(object_latents.shape[:-1], device=object_latents.device, dtype=object_latents.dtype), scale=1
    )
    object_latent_loss = (1 / shapes["k"]) * torch.distributions.kl.kl_divergence(object_latent_dist, latent_prior)
    # The KL doesn't reduce all the way because the distribution considers the batch size to be (batch, K, LATENT_CHANNELS)
    object_latent_loss = object_latent_loss.sum(dim=(2, 1))
    check_shape(object_latent_loss, "b", **shapes)
    # this is truly T not Td
    temporal_latent_loss = (1 / shapes["t"]) * torch.distributions.kl.kl_divergence(temporal_latent_dist, latent_prior)
    temporal_latent_loss = temporal_latent_loss.sum(dim=(2, 1))
    check_shape(temporal_latent_loss, "b", **shapes)

    return object_latent_loss, temporal_latent_loss
