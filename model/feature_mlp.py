import pytorch_lightning as pl
import torch
from einops import rearrange
from torch import Tensor

from constants import LATENT_CHANNELS
from constants import TRANSFORMER_CHANNELS
from constants import XY_SPATIAL_DIM_AFTER_TRANSFORMER
from constants import K
from constants import T
from model.common import MLP


class FeatureMLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.spatial_mlp = MLP(
            in_features=TRANSFORMER_CHANNELS, out_features=LATENT_CHANNELS * 2, hidden_features=[1024]
        )
        self.temporal_mlp = MLP(
            in_features=TRANSFORMER_CHANNELS, out_features=LATENT_CHANNELS * 2, hidden_features=[1024]
        )

    def forward(self, x: Tensor):
        x = rearrange(x, "b (t h w) c -> b t h w c", t=T, h=XY_SPATIAL_DIM_AFTER_TRANSFORMER, w=XY_SPATIAL_DIM_AFTER_TRANSFORMER, c=TRANSFORMER_CHANNELS)  # fmt: skip

        # Aggregate the temporal info to get the spatial features
        # Note that XY_SPATIAL_DIM_AFTER_TRANSFORMER ** 2 must equal K
        spatial = torch.mean(x, dim=1)
        # (b, h, w, c) at this point can also be interpreted as (b, k, c)
        spatial = rearrange(spatial, "b h w c -> (b h w) c")

        # Aggregate the spatial info to get the temporal features
        temporal = torch.mean(x, dim=(2, 3))
        temporal = rearrange(temporal, "b t c -> (b t) c", t=T, c=TRANSFORMER_CHANNELS)

        # Apply the MLPs
        spatial = self.spatial_mlp(spatial)
        # Reshape to have 2 channels, for (mean, log_scale)
        spatial = rearrange(spatial, "(b k) (c c2) -> b k c c2", k=K, c=LATENT_CHANNELS, c2=2)

        temporal = self.temporal_mlp(temporal)
        # Reshape to have 2 channels, for (mean, log_scale)
        temporal = rearrange(temporal, "(b t) (c c2) -> b t c c2", t=T, c=LATENT_CHANNELS, c2=2)

        return spatial, temporal
