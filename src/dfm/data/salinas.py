"""Salinas hyperspectral benchmark helpers.

The Salinas scene is a labeled AVIRIS hyperspectral benchmark. The corrected
version commonly used for classification contains 204 bands after removing
water absorption bands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np


SALINAS_CORRECTED_URLS = (
    "https://zenodo.org/records/15771735/files/Salinas_corrected.mat?download=1",
    "https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
)
SALINAS_GT_URLS = (
    "https://zenodo.org/records/15771735/files/Salinas_gt.mat?download=1",
    "https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
)

SALINAS_CLASS_NAMES: tuple[str, ...] = (
    "Brocoli_green_weeds_1",
    "Brocoli_green_weeds_2",
    "Fallow",
    "Fallow_rough_plow",
    "Fallow_smooth",
    "Stubble",
    "Celery",
    "Grapes_untrained",
    "Soil_vinyard_develop",
    "Corn_senesced_green_weeds",
    "Lettuce_romaine_4wk",
    "Lettuce_romaine_5wk",
    "Lettuce_romaine_6wk",
    "Lettuce_romaine_7wk",
    "Vinyard_untrained",
    "Vinyard_vertical_trellis",
)


@dataclass(frozen=True)
class SalinasScene:
    """Loaded Salinas cube and label map."""

    cube: np.ndarray
    labels: np.ndarray
    class_names: tuple[str, ...] = SALINAS_CLASS_NAMES

    @property
    def height(self) -> int:
        return int(self.cube.shape[0])

    @property
    def width(self) -> int:
        return int(self.cube.shape[1])

    @property
    def bands(self) -> int:
        return int(self.cube.shape[2])


def _download_with_fallback(urls: tuple[str, ...], output_path: Path) -> None:
    last_error: Exception | None = None
    for url in urls:
        try:
            urlretrieve(url, output_path)
            return
        except Exception as exc:  # noqa: BLE001 - preserve fallback details for user.
            last_error = exc
            if output_path.exists():
                output_path.unlink()
    raise RuntimeError(f"Failed to download {output_path.name}") from last_error


def download_salinas(data_dir: str | Path) -> tuple[Path, Path]:
    """Download Salinas corrected data and ground truth if missing."""

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    cube_path = data_dir / "Salinas_corrected.mat"
    gt_path = data_dir / "Salinas_gt.mat"

    if not cube_path.exists() or cube_path.stat().st_size < 1_000_000:
        print(f"Downloading {cube_path.name}...")
        _download_with_fallback(SALINAS_CORRECTED_URLS, cube_path)

    if not gt_path.exists() or gt_path.stat().st_size < 1_000:
        print(f"Downloading {gt_path.name}...")
        _download_with_fallback(SALINAS_GT_URLS, gt_path)

    return cube_path, gt_path


def load_salinas(data_dir: str | Path, download: bool = True) -> SalinasScene:
    """Load Salinas corrected cube and ground-truth labels."""

    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("Install scipy to read Salinas .mat files.") from exc

    data_dir = Path(data_dir)
    if download:
        cube_path, gt_path = download_salinas(data_dir)
    else:
        cube_path = data_dir / "Salinas_corrected.mat"
        gt_path = data_dir / "Salinas_gt.mat"

    if not cube_path.exists() or not gt_path.exists():
        raise FileNotFoundError(
            "Missing Salinas files. Run download_salinas(data_dir) first."
        )

    cube = loadmat(cube_path)["salinas_corrected"]
    labels = loadmat(gt_path)["salinas_gt"]

    return SalinasScene(cube=cube.astype(np.float32), labels=labels.astype(np.int64))


def normalize_cube_per_band(cube: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Standardize a hyperspectral cube band-by-band."""

    if cube.ndim != 3:
        raise ValueError(f"Expected cube with shape (H, W, C), got {cube.shape}.")
    cube = cube.astype(np.float32)
    mean = cube.mean(axis=(0, 1), keepdims=True)
    std = cube.std(axis=(0, 1), keepdims=True)
    return (cube - mean) / (std + eps)


def salinas_to_chw(scene: SalinasScene, normalize: bool = True) -> np.ndarray:
    """Convert Salinas from `(H, W, C)` to `(C, H, W)`."""

    cube = normalize_cube_per_band(scene.cube) if normalize else scene.cube.astype(np.float32)
    return cube.transpose(2, 0, 1)


def labeled_pixel_indices(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row, column, and zero-based class arrays for labeled pixels."""

    rows, cols = np.where(labels > 0)
    y = labels[rows, cols].astype(np.int64) - 1
    return rows, cols, y


def extract_labeled_center_patches(
    scene: SalinasScene,
    patch_size: int = 15,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract centered patches around every labeled pixel.

    Returns:
        patches: `(N, C, patch_size, patch_size)`
        labels: zero-based class labels with shape `(N,)`
    """

    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer.")

    cube = normalize_cube_per_band(scene.cube) if normalize else scene.cube.astype(np.float32)
    rows, cols, y = labeled_pixel_indices(scene.labels)
    radius = patch_size // 2
    padded = np.pad(cube, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")

    patches = np.empty((len(y), scene.bands, patch_size, patch_size), dtype=np.float32)
    for index, (row, col) in enumerate(zip(rows, cols)):
        patch = padded[row : row + patch_size, col : col + patch_size, :]
        patches[index] = patch.transpose(2, 0, 1)

    return patches, y


class SalinasPatchDataset:
    """Lazy centered-patch dataset for labeled Salinas pixels."""

    def __init__(
        self,
        scene: SalinasScene,
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


def stratified_train_test_split(
    labels: np.ndarray,
    train_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create stratified train/test indices without requiring scikit-learn."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []

    for class_id in np.unique(labels):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)
        train_count = max(1, int(round(len(class_indices) * train_fraction)))
        train_indices.append(class_indices[:train_count])
        test_indices.append(class_indices[train_count:])

    train = np.concatenate(train_indices)
    test = np.concatenate(test_indices)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test
