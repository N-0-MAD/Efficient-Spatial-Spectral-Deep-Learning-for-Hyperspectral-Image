"""Download the Salinas hyperspectral benchmark files."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfm.data.salinas import download_salinas


def main() -> None:
    data_dir = ROOT / "data" / "raw" / "salinas"
    cube_path, gt_path = download_salinas(data_dir)
    print(f"Cube: {cube_path}")
    print(f"Ground truth: {gt_path}")


if __name__ == "__main__":
    main()

