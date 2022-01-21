import pytorch_lightning as pl
import torch.nn.functional as F
from torch.nn import Conv2d
from einops import rearrange

from config import ENCODER_CONV_CHANNELS, T


class EncoderConv(pl.LightningModule):
    """The encoder is a simple stack of convolutional layers."""
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.c1 = Conv2d(in_channels=3, out_channels=ENCODER_CONV_CHANNELS, kernel_size=4, stride=2, padding=(1, 1))
        self.c2 = Conv2d(
            in_channels=ENCODER_CONV_CHANNELS,
            out_channels=ENCODER_CONV_CHANNELS,
            kernel_size=4,
            stride=2,
            padding=(1, 1),
        )
        self.c3 = Conv2d(
            in_channels=ENCODER_CONV_CHANNELS,
            out_channels=ENCODER_CONV_CHANNELS,
            kernel_size=4,
            stride=2,
            padding=(1, 1),
        )

    def forward(self, x):
        x = rearrange(x, "b t c h w -> (b t) c h w", t=T, c=3, h=64, w=64)

        x = self.c1(x)
        x = F.relu(x)

        x = self.c2(x)
        x = F.relu(x)

        x = self.c3(x)
        x = F.relu(x)

        x = rearrange(x, "(b t) c h w -> b t c h w", t=T, c=ENCODER_CONV_CHANNELS, w=8, h=8)

        return x
