"""Shared helpers for local hyperspectral benchmark datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


INDIAN_PINES_CLASS_NAMES: tuple[str, ...] = (
    "Alfalfa",
    "Corn-notill",
    "Corn-mintill",
    "Corn",
    "Grass-pasture",
    "Grass-trees",
    "Grass-pasture-mowed",
    "Hay-windrowed",
    "Oats",
    "Soybean-notill",
    "Soybean-mintill",
    "Soybean-clean",
    "Wheat",
    "Woods",
    "Buildings-Grass-Trees-Drives",
    "Stone-Steel-Towers",
)

PAVIA_U_CLASS_NAMES: tuple[str, ...] = (
    "Asphalt",
    "Meadows",
    "Gravel",
    "Trees",
    "Painted metal sheets",
    "Bare Soil",
    "Bitumen",
    "Self-Blocking Bricks",
    "Shadows",
)

BOTSWANA_CLASS_NAMES: tuple[str, ...] = (
    "Water",
    "Hippo grass",
    "Floodplain grasses 1",
    "Floodplain grasses 2",
    "Reeds",
    "Riparian",
    "Firescar",
    "Island interior",
    "Acacia woodlands",
    "Acacia shrublands",
    "Acacia grasslands",
    "Short mopane",
    "Mixed mopane",
    "Exposed soils",
)

KSC_CLASS_NAMES: tuple[str, ...] = (
    "Scrub",
    "Willow swamp",
    "CP hammock",
    "CP/Oak",
    "Slash pine",
    "Oak/Broadleaf",
    "Hardwood",
    "Swamp",
    "Graminoid marsh",
    "Spartina marsh",
    "Cattail marsh",
    "Salt marsh",
    "Mud flats",
)


@dataclass(frozen=True)
class HyperspectralScene:
    """Loaded hyperspectral cube and label map."""

    cube: np.ndarray
    labels: np.ndarray
    class_names: tuple[str, ...]
    name: str

    @property
    def height(self) -> int:
        return int(self.cube.shape[0])

    @property
    def width(self) -> int:
        return int(self.cube.shape[1])

    @property
    def bands(self) -> int:
        return int(self.cube.shape[2])


DATASET_SPECS = {
    "indian_pines": {
        "display_name": "Indian Pines",
        "cube_file": "Indian_pines_corrected.mat",
        "gt_file": "Indian_pines_gt.mat",
        "cube_key": "indian_pines_corrected",
        "gt_key": "indian_pines_gt",
        "class_names": INDIAN_PINES_CLASS_NAMES,
    },
    "pavia_u": {
        "display_name": "Pavia University",
        "cube_file": "PaviaU.mat",
        "gt_file": "PaviaU_gt.mat",
        "cube_key": "paviaU",
        "gt_key": "paviaU_gt",
        "class_names": PAVIA_U_CLASS_NAMES,
    },
    "botswana": {
        "display_name": "Botswana",
        "cube_file": "Botswana.mat",
        "gt_file": "Botswana_gt.mat",
        "cube_key": "Botswana",
        "gt_key": "Botswana_gt",
        "class_names": BOTSWANA_CLASS_NAMES,
    },
    "ksc": {
        "display_name": "KSC",
        "cube_file": "KSC.mat",
        "gt_file": "KSC_gt.mat",
        "cube_key": "KSC",
        "gt_key": "KSC_gt",
        "class_names": KSC_CLASS_NAMES,
    },
}


def _resolve_data_dir(data_dir: str | Path, dataset_name: str) -> Path:
    data_dir = Path(data_dir)
    candidates = (
        data_dir,
        data_dir / dataset_name,
        data_dir / "raw" / dataset_name,
    )

    spec = DATASET_SPECS[dataset_name]
    for candidate in candidates:
        if (candidate / spec["cube_file"]).exists() and (candidate / spec["gt_file"]).exists():
            return candidate

    expected = " or ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Missing {spec['display_name']} files. Expected {spec['cube_file']} "
        f"and {spec['gt_file']} in {expected}."
    )


def _mat_array(mat: dict, preferred_key: str, expected_ndim: int, file_name: str) -> np.ndarray:
    if preferred_key in mat:
        return mat[preferred_key]

    candidates = [
        value
        for key, value in mat.items()
        if not key.startswith("__")
        and isinstance(value, np.ndarray)
        and value.ndim == expected_ndim
    ]

    if len(candidates) == 1:
        return candidates[0]

    available_keys = [key for key in mat if not key.startswith("__")]
    raise KeyError(
        f"Could not find key {preferred_key!r} in {file_name}. "
        f"Available keys: {available_keys}"
    )


def load_hyperspectral_benchmark(
    dataset_name: str,
    data_dir: str | Path,
) -> HyperspectralScene:
    """Load a supported local benchmark dataset from MATLAB files."""

    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("Install scipy to read hyperspectral .mat files.") from exc

    if dataset_name not in DATASET_SPECS:
        raise KeyError(f"Unsupported dataset: {dataset_name}")

    spec = DATASET_SPECS[dataset_name]
    resolved_dir = _resolve_data_dir(data_dir, dataset_name)

    cube_mat = loadmat(resolved_dir / spec["cube_file"])
    gt_mat = loadmat(resolved_dir / spec["gt_file"])

    cube = _mat_array(cube_mat, spec["cube_key"], expected_ndim=3, file_name=spec["cube_file"])
    labels = _mat_array(gt_mat, spec["gt_key"], expected_ndim=2, file_name=spec["gt_file"])

    return HyperspectralScene(
        cube=cube.astype(np.float32),
        labels=labels.astype(np.int64),
        class_names=spec["class_names"],
        name=spec["display_name"],
    )


def normalize_cube_per_band(cube: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Standardize a hyperspectral cube band-by-band."""

    if cube.ndim != 3:
        raise ValueError(f"Expected cube with shape (H, W, C), got {cube.shape}.")

    cube = cube.astype(np.float32)
    mean = cube.mean(axis=(0, 1), keepdims=True)
    std = cube.std(axis=(0, 1), keepdims=True)
    return (cube - mean) / (std + eps)


def labeled_pixel_indices(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row, column, and zero-based class arrays for labeled pixels."""

    rows, cols = np.where(labels > 0)
    y = labels[rows, cols].astype(np.int64) - 1
    return rows, cols, y


class HyperspectralPatchDataset:
    """Lazy centered-patch dataset for labeled hyperspectral pixels."""

    def __init__(
        self,
        scene: HyperspectralScene,
        indices: np.ndarray | None = None,
        patch_size: int = 15,
        normalize: bool = True,
    ) -> None:
        if patch_size <= 0 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer.")

        rows, cols, labels = labeled_pixel_indices(scene.labels)
        if indices is None:
            indices = np.arange(len(labels))

        self.rows = rows[indices]
        self.cols = cols[indices]
        self.labels = labels[indices]
        self.patch_size = patch_size
        self.radius = patch_size // 2

        cube = normalize_cube_per_band(scene.cube) if normalize else scene.cube.astype(np.float32)
        self.padded_cube = np.pad(
            cube,
            ((self.radius, self.radius), (self.radius, self.radius), (0, 0)),
            mode="reflect",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.int64]:
        row = self.rows[index]
        col = self.cols[index]
        patch = self.padded_cube[
            row : row + self.patch_size,
            col : col + self.patch_size,
            :,
        ]
        return patch.transpose(2, 0, 1).astype(np.float32), self.labels[index]
