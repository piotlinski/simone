import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer

from constants import ENCODER_CONV_CHANNELS
from constants import TRANSFORMER_CHANNELS
from constants import XY_SPATIAL_DIM_AFTER_CONV_ENCODER
from constants import XY_SPATIAL_DIM_AFTER_TRANSFORMER
from constants import T
from contrib.position_encoding import PositionalEncoding3D


class EncoderTransformer(pl.LightningModule):
    def __init__(self, transformer_layers: int):
        super().__init__()
        self.save_hyperparameters()

        # this template layer will get cloned inside the TransformerEncoder modules below.
        encoder_layer_template = TransformerEncoderLayer(
            d_model=TRANSFORMER_CHANNELS,
            nhead=5,
            dim_feedforward=1024,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.linear_layer = torch.nn.Linear(
            in_features=ENCODER_CONV_CHANNELS, out_features=TRANSFORMER_CHANNELS, bias=False
        )

        self.transformer_1 = TransformerEncoder(encoder_layer=encoder_layer_template, num_layers=transformer_layers)
        self.transformer_2 = TransformerEncoder(encoder_layer=encoder_layer_template, num_layers=transformer_layers)

        self.position_encoding_1 = PositionalEncoding3D(TRANSFORMER_CHANNELS)
        self.position_encoding_2 = PositionalEncoding3D(TRANSFORMER_CHANNELS)

    def forward(self, x: Tensor):
        batch_size = x.shape[0]
        x = rearrange(x, "b t c h w -> b t h w c", b=batch_size, t=T, h=XY_SPATIAL_DIM_AFTER_CONV_ENCODER, w=XY_SPATIAL_DIM_AFTER_CONV_ENCODER, c=ENCODER_CONV_CHANNELS)  # fmt: skip

        # apply linear transformation to project ENCODER_CONV_CHANNELS to TRANSFORMER_CHANNELS
        x = self.linear_layer(x)

        # apply 3d position encoding before going through the first transformer
        x = x + self.position_encoding_1(x)
        x = rearrange(x, "b t h w c -> b (t h w) c", b=batch_size, t=T, h=XY_SPATIAL_DIM_AFTER_CONV_ENCODER, w=XY_SPATIAL_DIM_AFTER_CONV_ENCODER, c=TRANSFORMER_CHANNELS)  # fmt: skip

        x = self.transformer_1(x)
        x = rearrange(x, "b (t h w) c -> (b t) c h w", b=batch_size, t=T, h=XY_SPATIAL_DIM_AFTER_CONV_ENCODER, w=XY_SPATIAL_DIM_AFTER_CONV_ENCODER, c=TRANSFORMER_CHANNELS)  # fmt: skip
        # This is the scaling that the paper suggests,
        # (K / XY_SPATIAL_DIM_AFTER_CONV_ENCODER**2) ** 0.5 equals 1/2 with default values
        # x = F.avg_pool2d(x, kernel_size=2) * (K / XY_SPATIAL_DIM_AFTER_CONV_ENCODER**2) ** 0.5
        # But I found that this works notably better, at least early in training.
        x = F.avg_pool2d(x, kernel_size=2) * 2
        x = rearrange(x, "(b t) c h w -> b t h w c", b=batch_size, t=T, h=XY_SPATIAL_DIM_AFTER_TRANSFORMER, w=XY_SPATIAL_DIM_AFTER_TRANSFORMER, c=TRANSFORMER_CHANNELS)  # fmt: skip

        # add another 3d position encoding before the second transformer
        x = x + self.position_encoding_2(x)
        x = rearrange(x, "b t h w c -> b (t h w) c", b=batch_size, t=T, h=XY_SPATIAL_DIM_AFTER_TRANSFORMER, w=XY_SPATIAL_DIM_AFTER_TRANSFORMER, c=TRANSFORMER_CHANNELS)  # fmt: skip
        x = self.transformer_2(x)
        assert x.shape == (batch_size, T * XY_SPATIAL_DIM_AFTER_TRANSFORMER * XY_SPATIAL_DIM_AFTER_TRANSFORMER, TRANSFORMER_CHANNELS)  # fmt: skip

        return x
