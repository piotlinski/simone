import random
import argparse

import numpy as np
import torch
import torch.nn as nn


def check_shape(x, shape, **kwargs):
    """An einops-style shape assertion.

    Discussed in https://github.com/arogozhnikov/einops/issues/168
    """
    dims = shape.split(" ")
    assert len(x.shape) == len(dims)
    for k, v in kwargs.items():
        # assert k in dims
        assert x.shape[dims.index(k)] == v



def get_latent_dist(latent, log_scale_min=-10, log_scale_max=3):
    """Convert the MLP output (with mean and log std) into a torch `Normal` distribution."""
    means = latent[..., 0]
    log_scale = latent[..., 1]
    # Clamp the minimum to keep latents from getting too far into the saturating region of the exp
    # And the max because i noticed it exploding early in the training sometimes
    log_scale = log_scale.clamp(min=log_scale_min, max=log_scale_max)
    dist = torch.distributions.normal.Normal(means, torch.exp(log_scale))
    return dist


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def generate_color_palette(num_masks: int):
    out = []
    for i in range(num_masks):
        out.append(tuple([random.randint(0, 255) for _ in range(3)]))
    return torch.tensor(out).to(torch.float) / 255


def generate_segmentation(weights, colors):
    colors = colors.to(weights.device)
    # weights should have shape b, t, k, h, w
    b, t, k, h, w = weights.shape
    assert len(colors) == k
    # colors should have shape k, 3
    ce = colors.view(1, 1, k, 1, 1, 3).expand(b, t, k, h, w, 3)
    wa = weights.argmax(dim=2)
    we = wa.view(b, t, 1, h, w, 1).expand(b, t, 1, h, w, 3)
    return torch.gather(ce, 2, we).view(b, t, h, w, 3)


# from https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/positional_encodings.py
class PositionalEncoding3D(nn.Module):
    def __init__(self, channels, freq_base=1.0, freq_scale=10000.0):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = freq_base / (freq_scale ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)

        The encodings for each of (x, y, z) are stacked in the channel dim.
        So the first ch/3 channels are the x encodings.
        For each of x, y, z, the sin, cos embeddings are also stacked in the channel dim.
        So the first ch/6 are x_sin, and the second ch/6 are x_cos.
        They start at high freq and go to low freq.
        So ch0 is the highest freq x sin encoding.
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(tensor.type())
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        return emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)


def compute_cov_matrix(X):
    # torch.cov doesn't support batched cov calculation :/
    D = X.shape[-1]
    mean = torch.mean(X, dim=-1).unsqueeze(-1)
    X = X - mean
    return 1 / (D - 1) * X @ X.transpose(-1, -2)
