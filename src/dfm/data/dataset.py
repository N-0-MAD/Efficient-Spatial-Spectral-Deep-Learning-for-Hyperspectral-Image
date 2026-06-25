"""Small dataset containers shared by notebooks and training scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PatchRecord:
    """One model-ready remote sensing patch plus metadata."""

    patch: np.ndarray
    label: int | None
    sensor: str
    band_names: tuple[str, ...]
    metadata: dict[str, Any]


class InMemoryPatchDataset:
    """Simple NumPy dataset for early experiments and notebook prototypes."""

    def __init__(
        self,
        patches: np.ndarray,
        labels: np.ndarray | None = None,
        sensor: str = "unknown",
        band_names: tuple[str, ...] = (),
    ) -> None:
        if patches.ndim != 4:
            raise ValueError(
                f"Expected patches with shape (N, C, H, W), got {patches.shape}."
            )
        if labels is not None and len(labels) != len(patches):
            raise ValueError("labels must have the same length as patches.")

        self.patches = patches
        self.labels = labels
        self.sensor = sensor
        self.band_names = band_names

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> PatchRecord:
        label = None if self.labels is None else int(self.labels[index])
        return PatchRecord(
            patch=self.patches[index],
            label=label,
            sensor=self.sensor,
            band_names=self.band_names,
            metadata={"index": index},
        )

