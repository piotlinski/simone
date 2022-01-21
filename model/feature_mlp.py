import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange

from config import TRANSFORMER_CHANNELS, LATENT_CHANNELS, K, T


class FeatureMLP(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.spatial_linear_1 = torch.nn.Linear(in_features=TRANSFORMER_CHANNELS, out_features=1024)
        self.spatial_linear_2 = torch.nn.Linear(in_features=1024, out_features=LATENT_CHANNELS * 2)
        self.temporal_linear_1 = torch.nn.Linear(in_features=TRANSFORMER_CHANNELS, out_features=1024)
        self.temporal_linear_2 = torch.nn.Linear(in_features=1024, out_features=LATENT_CHANNELS * 2)

    def forward(self, x):
        # x is the output of the transformer
        x = rearrange(x, "b (t h w) c -> b t h w c", t=T, h=4, w=4, c=TRANSFORMER_CHANNELS)

        # Aggregate the temporal info to get the spatial features
        spatial = torch.mean(x, dim=1)
        spatial = rearrange(spatial, "b h w c -> (b h w) c")

        # Aggregate the spatial info to get the temporal features
        temporal = torch.mean(x, dim=(2, 3))
        temporal = rearrange(temporal, "b t c -> (b t) c", t=T, c=TRANSFORMER_CHANNELS)

        # Apply the MLPs
        spatial = self.spatial_linear_1(spatial)
        spatial = F.relu(spatial)
        spatial = self.spatial_linear_2(spatial)
        # Reshape to have 2 channels, for (mean, log_scale)
        spatial = rearrange(spatial, "(b k) (c c2) -> b k c c2", k=K, c=LATENT_CHANNELS, c2=2)

        temporal = self.temporal_linear_1(temporal)
        temporal = F.relu(temporal)
        temporal = self.temporal_linear_2(temporal)
        # Reshape to have 2 channels, for (mean, log_scale)
        temporal = rearrange(temporal, "(b t) (c c2) -> b t c c2", t=T, c=LATENT_CHANNELS, c2=2)

        return spatial, temporal
