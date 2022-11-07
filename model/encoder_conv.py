from functools import partial

import pytorch_lightning as pl
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn import Conv2d

from constants import CONV_STRIDE
from constants import ENCODER_CONV_CHANNELS
from constants import XY_RESOLUTION
from constants import XY_SPATIAL_DIM_AFTER_CONV_ENCODER
from constants import T


class EncoderConv(pl.LightningModule):
    """The encoder is a simple stack of convolutional layers."""

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        conv_layer = partial(Conv2d, kernel_size=4, stride=CONV_STRIDE, padding=(1, 1))
        self.conv_1 = conv_layer(in_channels=3, out_channels=ENCODER_CONV_CHANNELS)
        self.conv_2 = conv_layer(in_channels=ENCODER_CONV_CHANNELS, out_channels=ENCODER_CONV_CHANNELS)
        self.conv_3 = conv_layer(in_channels=ENCODER_CONV_CHANNELS, out_channels=ENCODER_CONV_CHANNELS)
        self.conv_4 = conv_layer(in_channels=ENCODER_CONV_CHANNELS, out_channels=ENCODER_CONV_CHANNELS)

    def forward(self, x: Tensor):
        x = rearrange(x, "b t c h w -> (b t) c h w", t=T, c=3, h=XY_RESOLUTION, w=XY_RESOLUTION)

        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = F.relu(self.conv_4(x))
        return rearrange(x, "(b t) c h w -> b t c h w", t=T, c=ENCODER_CONV_CHANNELS, w=XY_SPATIAL_DIM_AFTER_CONV_ENCODER, h=XY_SPATIAL_DIM_AFTER_CONV_ENCODER)  # fmt: skip
