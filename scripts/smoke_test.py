"""Local smoke test for shape-safe preprocessing utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfm.data.patching import extract_patches
from dfm.experiments.token_scaling import (
    attention_cost_proxy,
    grouped_band_tokens,
    independent_band_tokens,
)


def main() -> None:
    image = np.random.default_rng(42).normal(size=(4, 1121, 851)).astype(np.float32)
    patches, grid = extract_patches(image, patch_size=32)

    assert grid.cropped_height == 1120
    assert grid.cropped_width == 832
    assert patches.shape == (910, 4, 32, 32)

    independent = independent_band_tokens(
        height=grid.cropped_height,
        width=grid.cropped_width,
        bands=12,
        patch_size=32,
    )
    grouped = grouped_band_tokens(
        height=grid.cropped_height,
        width=grid.cropped_width,
        bands=12,
        patch_size=32,
        group_size=4,
    )

    assert independent == 910 * 12
    assert grouped == 910 * 3

    print("Smoke test passed.")
    print(f"Patches: {patches.shape}")
    print(f"Independent-band tokens for 12 bands: {independent:,}")
    print(f"Grouped-band tokens for 12 bands, group_size=4: {grouped:,}")
    print(f"Independent attention proxy: {attention_cost_proxy(independent):,}")


if __name__ == "__main__":
    main()

