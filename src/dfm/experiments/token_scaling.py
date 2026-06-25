"""Token-count and attention-cost proxy calculations."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class TokenScalingResult:
    strategy: str
    height: int
    width: int
    bands: int
    patch_size: int
    tokens: int
    attention_cost_proxy: int


def spatial_patch_count(height: int, width: int, patch_size: int, ceil_edges: bool = False) -> int:
    """Count spatial patches for a patch grid."""

    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if ceil_edges:
        return ceil(height / patch_size) * ceil(width / patch_size)
    return (height // patch_size) * (width // patch_size)


def independent_band_tokens(height: int, width: int, bands: int, patch_size: int) -> int:
    """Tokens when every band-patch is independent."""

    return spatial_patch_count(height, width, patch_size) * bands


def all_band_patch_tokens(height: int, width: int, patch_size: int) -> int:
    """Tokens when all bands inside a spatial patch are embedded together."""

    return spatial_patch_count(height, width, patch_size)


def grouped_band_tokens(height: int, width: int, bands: int, patch_size: int, group_size: int) -> int:
    """Tokens when bands are grouped before tokenization."""

    if group_size <= 0:
        raise ValueError("group_size must be positive.")
    groups = ceil(bands / group_size)
    return spatial_patch_count(height, width, patch_size) * groups


def attention_cost_proxy(tokens: int) -> int:
    """Quadratic self-attention cost proxy."""

    return tokens * tokens


def compare_tokenization(
    height: int,
    width: int,
    band_counts: list[int],
    patch_sizes: list[int],
    group_size: int = 8,
) -> list[TokenScalingResult]:
    """Build a comparison table for common tokenization strategies."""

    results: list[TokenScalingResult] = []
    for bands in band_counts:
        for patch_size in patch_sizes:
            strategies = {
                "independent_band": independent_band_tokens(height, width, bands, patch_size),
                "all_band_patch": all_band_patch_tokens(height, width, patch_size),
                f"grouped_band_{group_size}": grouped_band_tokens(
                    height,
                    width,
                    bands,
                    patch_size,
                    group_size,
                ),
            }
            for strategy, tokens in strategies.items():
                results.append(
                    TokenScalingResult(
                        strategy=strategy,
                        height=height,
                        width=width,
                        bands=bands,
                        patch_size=patch_size,
                        tokens=tokens,
                        attention_cost_proxy=attention_cost_proxy(tokens),
                    )
                )
    return results

