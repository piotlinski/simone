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


class EncoderTransformer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        l = TransformerEncoderLayer(
            d_model=TRANSFORMER_CHANNELS,
            nhead=5,
            dim_feedforward=1024,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.l1 = torch.nn.Linear(in_features=ENCODER_CONV_CHANNELS, out_features=TRANSFORMER_CHANNELS, bias=False)

        self.t1 = TransformerEncoder(encoder_layer=l, num_layers=self.hparams.transformer_layers)
        self.t2 = TransformerEncoder(encoder_layer=l, num_layers=self.hparams.transformer_layers)

        self.p1 = PositionalEncoding3D(TRANSFORMER_CHANNELS)
        self.p2 = PositionalEncoding3D(TRANSFORMER_CHANNELS)

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "b t c h w -> b t h w c", t=T, h=8, w=8, c=ENCODER_CONV_CHANNELS)

        # Apply linear transformation to project ENCODER_CONV_CHANNELS to TRANSFORMER_CHANNELS
        x = self.l1(x)

        # apply 3d position encoding
        x = x + self.p1(x)

        # Convert TxWxH into a linear sequence for the transformer
        x = rearrange(x, "b t h w c -> b (t h w) c", t=T, h=8, w=8, c=TRANSFORMER_CHANNELS)

        # First transformer stage
        x = self.t1(x)

        # Do their weird pooling to reshape from TxWxH to TxK
        # Need to convert to b * t, c, h, w
        x = rearrange(x, "b (t h w) c -> (b t) c h w", t=T, h=8, w=8, c=TRANSFORMER_CHANNELS)
        x = F.avg_pool2d(x, kernel_size=2) * 4 / 2
        x = rearrange(x, "(b t) c h w -> b t h w c", t=T, h=4, w=4, c=TRANSFORMER_CHANNELS)

        # Another position encoding
        x = x + self.p2(x)

        x = rearrange(x, "b t h w c -> b (t h w) c", t=T, h=4, w=4, c=TRANSFORMER_CHANNELS)
        # Second transformer stage
        x = self.t2(x)
        assert x.shape == (batch_size, T * 4 * 4, TRANSFORMER_CHANNELS)
        return x
