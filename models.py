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
