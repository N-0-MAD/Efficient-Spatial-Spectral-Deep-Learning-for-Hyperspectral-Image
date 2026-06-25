"""Normalization helpers for reflectance-like tensors."""

from __future__ import annotations

import numpy as np


def scale_reflectance(
    image: np.ndarray,
    scale: float = 10_000.0,
    clip: bool = True,
) -> np.ndarray:
    """Scale integer reflectance values to approximately `[0, 1]`."""

    scaled = image.astype(np.float32) / float(scale)
    if clip:
        scaled = np.clip(scaled, 0.0, 1.0)
    return scaled


def standardize_per_band(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Standardize each spectral band independently."""

    if image.ndim != 3:
        raise ValueError(f"Expected image with shape (C, H, W), got {image.shape}.")

    image = image.astype(np.float32)
    mean = image.mean(axis=(1, 2), keepdims=True)
    std = image.std(axis=(1, 2), keepdims=True)
    return (image - mean) / (std + eps)


def rgb_from_bands(
    image: np.ndarray,
    band_names: list[str] | tuple[str, ...],
    rgb_order: tuple[str, str, str] = ("B04", "B03", "B02"),
    display_scale: float | None = 3_000.0,
    percentile_stretch: bool = False,
    percentiles: tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """Create a Matplotlib-ready RGB image from a channel-first band stack."""

    if image.ndim != 3:
        raise ValueError(f"Expected image with shape (C, H, W), got {image.shape}.")

    band_to_index = {band: idx for idx, band in enumerate(band_names)}
    missing = [band for band in rgb_order if band not in band_to_index]
    if missing:
        raise ValueError(f"Missing RGB bands: {missing}.")

    rgb = np.stack([image[band_to_index[band]] for band in rgb_order], axis=0)
    rgb = rgb.astype(np.float32)

    if percentile_stretch:
        stretched = np.zeros_like(rgb, dtype=np.float32)
        for channel_index in range(rgb.shape[0]):
            channel = rgb[channel_index]
            low, high = np.nanpercentile(channel, percentiles)
            if high <= low:
                continue
            stretched[channel_index] = np.clip((channel - low) / (high - low), 0.0, 1.0)
        rgb = stretched
    elif display_scale is not None:
        rgb = np.clip(rgb / display_scale, 0.0, 1.0)
    else:
        rgb = np.clip(rgb, 0.0, 1.0)

    return rgb.transpose(1, 2, 0)
