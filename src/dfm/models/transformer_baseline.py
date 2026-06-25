"""Independent-band transformer baseline.

This intentionally mirrors the bottleneck under study: every spatial patch in
every band becomes its own token, so sequence length grows as patches * bands.
"""

from __future__ import annotations

import torch
from torch import nn


class IndependentBandPatchEmbed(nn.Module):
    """Convert `(B, C, H, W)` into independent band-patch tokens."""

    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        max_bands: int = 256,
        max_patches: int = 4096,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed = nn.Linear(patch_size * patch_size, embed_dim)
        self.band_embedding = nn.Embedding(max_bands, embed_dim)
        self.patch_embedding = nn.Embedding(max_patches, embed_dim)
        self.max_bands = max_bands
        self.max_patches = max_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        patch_size = self.patch_size

        if channels > self.max_bands:
            raise ValueError(f"channels={channels} exceeds max_bands={self.max_bands}.")
        if height < patch_size or width < patch_size:
            raise ValueError("Input is smaller than patch_size.")

        cropped_h = (height // patch_size) * patch_size
        cropped_w = (width // patch_size) * patch_size
        x = x[:, :, :cropped_h, :cropped_w]

        patches_h = cropped_h // patch_size
        patches_w = cropped_w // patch_size
        num_spatial_patches = patches_h * patches_w
        if num_spatial_patches > self.max_patches:
            raise ValueError(
                f"num_spatial_patches={num_spatial_patches} exceeds "
                f"max_patches={self.max_patches}."
            )

        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(
            batch,
            channels,
            num_spatial_patches,
            patch_size * patch_size,
        )
        tokens = self.embed(patches)

        band_ids = torch.arange(channels, device=x.device).view(1, channels, 1)
        patch_ids = torch.arange(num_spatial_patches, device=x.device).view(
            1,
            1,
            num_spatial_patches,
        )
        tokens = tokens + self.band_embedding(band_ids) + self.patch_embedding(patch_ids)
        return tokens.view(batch, channels * num_spatial_patches, -1)


class IndependentBandTransformerClassifier(nn.Module):
    """Full-attention baseline over independent band-patch tokens."""

    def __init__(
        self,
        num_classes: int,
        patch_size: int = 16,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_bands: int = 256,
        max_patches: int = 4096,
    ) -> None:
        super().__init__()
        self.patch_embed = IndependentBandPatchEmbed(
            patch_size=patch_size,
            embed_dim=embed_dim,
            max_bands=max_bands,
            max_patches=max_patches,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tokens)
        return self.head(self.norm(encoded[:, 0]))
