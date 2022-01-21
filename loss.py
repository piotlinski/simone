import torch
from einops import repeat

from config import K


def pixel_likelihood_loss(pixels, target, weights_softmax, sigma_x):
    # Expand the target in the object dimension, so it matches the shape of `pixels`
    target = repeat(target, "b t h w c -> b t k h w c", k=K)
    # Compute the log prob of each predicted pixel, for all object channels
    log_prob = torch.distributions.normal.Normal(pixels, sigma_x).log_prob(target)

    # assert log_prob.shape == (batch_size, Td, K, xy_dim, xy_dim, 3)

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
