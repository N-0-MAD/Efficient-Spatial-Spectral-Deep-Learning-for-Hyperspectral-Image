"""Generate a token-scaling comparison table."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfm.experiments.token_scaling import compare_tokenization


def main() -> None:
    height = 1120
    width = 832
    band_counts = [4, 12, 64, 128, 224]
    patch_sizes = [8, 16, 32, 64]
    results = compare_tokenization(
        height=height,
        width=width,
        band_counts=band_counts,
        patch_sizes=patch_sizes,
        group_size=8,
    )

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "token_scaling.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "strategy",
                "height",
                "width",
                "bands",
                "patch_size",
                "tokens",
                "attention_cost_proxy",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)

    print(f"Wrote {output_path}")
    print()
    print("strategy,bands,patch_size,tokens,attention_cost_proxy")
    for row in results:
        if row.patch_size == 32:
            print(
                f"{row.strategy},{row.bands},{row.patch_size},"
                f"{row.tokens},{row.attention_cost_proxy}"
            )


if __name__ == "__main__":
    main()

