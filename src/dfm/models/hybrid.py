"""Hybrid spatial-spectral classifier."""

from __future__ import annotations

import torch
from torch import nn


class SpatialCNNBranch(nn.Module):
    """Efficient local spatial branch."""

    def __init__(self, in_channels: int, feature_dim: int = 128) -> None:
        super().__init__()
        hidden = max(feature_dim // 2, 32)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectralAttentionBranch(nn.Module):
    """Attention across spectral bands after spatial pooling."""

    def __init__(
        self,
        feature_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        max_bands: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.value_projection = nn.Linear(1, feature_dim)
        self.band_embedding = nn.Embedding(max_bands, feature_dim)
        self.max_bands = max_bands
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        if channels > self.max_bands:
            raise ValueError(f"channels={channels} exceeds max_bands={self.max_bands}.")

        band_values = x.mean(dim=(-2, -1)).unsqueeze(-1)
        tokens = self.value_projection(band_values)
        band_ids = torch.arange(channels, device=x.device).view(1, channels)
        tokens = tokens + self.band_embedding(band_ids)
        encoded = self.encoder(tokens)
        return self.norm(encoded.mean(dim=1))


class GatedFusion(nn.Module):
    """Adaptive fusion between spatial and spectral features."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, spatial: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([spatial, spectral], dim=-1))
        return gate * spatial + (1.0 - gate) * spectral


class HybridSpatialSpectralClassifier(nn.Module):
    """CNN spatial branch + spectral attention branch + adaptive fusion."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        feature_dim: int = 128,
        spectral_depth: int = 2,
        spectral_heads: int = 4,
        fusion: str = "gated",
        dropout: float = 0.1,
        max_bands: int = 256,
    ) -> None:
        super().__init__()
        if fusion not in {"concat", "gated"}:
            raise ValueError("fusion must be either 'concat' or 'gated'.")

        self.fusion = fusion
        self.spatial = SpatialCNNBranch(in_channels=in_channels, feature_dim=feature_dim)
        self.spectral = SpectralAttentionBranch(
            feature_dim=feature_dim,
            depth=spectral_depth,
            num_heads=spectral_heads,
            dropout=dropout,
            max_bands=max_bands,
        )
        if fusion == "gated":
            self.fuser = GatedFusion(feature_dim)
            head_in = feature_dim
        else:
            self.fuser = nn.Identity()
            head_in = feature_dim * 2

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial(x)
        spectral = self.spectral(x)
        if self.fusion == "gated":
            fused = self.fuser(spatial, spectral)
        else:
            fused = torch.cat([spatial, spectral], dim=-1)
        return self.head(fused)
