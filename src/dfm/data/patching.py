"""Patch extraction utilities for channel-first remote sensing tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PatchGrid:
    """Metadata describing a non-overlapping patch grid."""

    channels: int
    original_height: int
    original_width: int
    cropped_height: int
    cropped_width: int
    patch_size: int

    @property
    def patches_h(self) -> int:
        return self.cropped_height // self.patch_size

    @property
    def patches_w(self) -> int:
        return self.cropped_width // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.patches_h * self.patches_w


def _validate_chw(image: np.ndarray) -> None:
    if image.ndim != 3:
        raise ValueError(f"Expected image with shape (C, H, W), got {image.shape}.")


def patch_grid(image: np.ndarray, patch_size: int) -> PatchGrid:
    """Compute the exact non-overlapping patch grid after edge cropping."""

    _validate_chw(image)
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")

    channels, height, width = image.shape
    cropped_height = (height // patch_size) * patch_size
    cropped_width = (width // patch_size) * patch_size

    if cropped_height == 0 or cropped_width == 0:
        raise ValueError(
            f"Patch size {patch_size} is too large for image shape {image.shape}."
        )

    return PatchGrid(
        channels=channels,
        original_height=height,
        original_width=width,
        cropped_height=cropped_height,
        cropped_width=cropped_width,
        patch_size=patch_size,
    )


def crop_to_patch_multiple(image: np.ndarray, patch_size: int) -> tuple[np.ndarray, PatchGrid]:
    """Crop channel-first imagery so H and W are exact multiples of patch_size."""

    grid = patch_grid(image, patch_size)
    cropped = image[:, : grid.cropped_height, : grid.cropped_width]
    return cropped, grid


def extract_patches(image: np.ndarray, patch_size: int) -> tuple[np.ndarray, PatchGrid]:
    """Extract non-overlapping patches from a `(C, H, W)` tensor.

    Returns patches with shape `(N, C, patch_size, patch_size)`.
    """

    cropped, grid = crop_to_patch_multiple(image, patch_size)
    reshaped = cropped.reshape(
        grid.channels,
        grid.patches_h,
        patch_size,
        grid.patches_w,
        patch_size,
    )
    patches = reshaped.transpose(1, 3, 0, 2, 4).reshape(
        grid.num_patches,
        grid.channels,
        patch_size,
        patch_size,
    )
    return patches, grid

