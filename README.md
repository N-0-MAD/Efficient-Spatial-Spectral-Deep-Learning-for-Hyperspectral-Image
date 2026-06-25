# Dense Forest Monitoring

Research starter repo for **Selective Spectral Attention with Efficient Spatial Modeling for Scalable Remote Sensing Foundation Models**.

The project studies one core bottleneck: independent band tokenization creates `#spatial_patches * #spectral_bands` tokens, and transformer attention becomes expensive as token count grows. The proposed direction is to process spatial structure with efficient local operators and reserve attention for compact spectral relationships.

## Research Objective

Build a remote sensing model that:

1. scales better than full independent-band tokenization,
2. is more efficient than full attention everywhere,
3. preserves useful spectral information for dense forest monitoring.

## Pipeline

1. **Data ingestion:** load Sentinel-2 L2A or hyperspectral scenes as `(C, H, W)` tensors.
2. **Preprocessing:** cloud/no-data mask, normalize bands, resample to a common resolution, and tile into fixed patches.
3. **Token scaling analysis:** compare token counts and attention-cost proxies across tokenization strategies.
4. **Baselines:** train CNN-only, full transformer, and simple hybrid baselines.
5. **Proposed model:** use a CNN/local spatial branch, spectral attention branch, and fusion module.
6. **Evaluation:** report accuracy, F1, parameters, FLOPs, peak memory, inference time, and scaling behavior as band count grows.
7. **Ablation:** test tokenization, patch size, fusion type, and attention placement.

## Project Layout

```text
dense-forest-monitoring/
  configs/
    sentinel2_redwood.yaml
  notebooks/
    01_sentinel2_data_loading.ipynb
    02_token_scaling_analysis.ipynb
    03_baseline_cnn.ipynb
    04_hybrid_model.ipynb
    05_salinas_cnn_training.ipynb
    06_salinas_hybrid_training.ipynb
    07_efficiency_comparison.ipynb
    08_salinas_transformer_training.ipynb
  scripts/
    run_token_scaling.py
    smoke_test.py
  src/dfm/
    data/
      patching.py
      sentinel2.py
      transforms.py
      dataset.py
    experiments/
      token_scaling.py
    models/
      cnn_baseline.py
      transformer_baseline.py
      hybrid.py
```

## Quick Start

Create and activate a local virtual environment:

```bash
python -m venv .venv
```

On PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Command Prompt:

```bat
.\.venv\Scripts\activate.bat
```

Then install dependencies:

```bash
pip install -e .
pip install -r requirements.txt
```

Run the local NumPy smoke test:

```bash
python scripts/smoke_test.py
```

Run token scaling analysis:

```bash
python scripts/run_token_scaling.py
```

Download Salinas:

```bash
python scripts/download_salinas.py
```

The Sentinel-2 notebook needs internet access and STAC dependencies, so Colab is currently the easiest environment for that step.

For dataset choices beyond Sentinel-2, see [DATA_SOURCES.md](DATA_SOURCES.md).

## Core References

- [FoMo-Bench / FoMo-Net](https://arxiv.org/abs/2312.10114): forest monitoring benchmark and flexible remote sensing foundation model framing.
- [SatMAE](https://arxiv.org/abs/2207.08051): multispectral and temporal masked autoencoding with spectral positional encodings.
- [SpectralGPT](https://arxiv.org/abs/2311.07113): spectral remote sensing foundation model.
- [SpectralMAE](https://www.mdpi.com/2231356): spectral masked autoencoder for hyperspectral reconstruction.
- [HybridSN](https://arxiv.org/abs/1902.06701): 3D-2D CNN baseline for hyperspectral image classification.
- [SpectralFormer](https://arxiv.org/abs/2107.02988): transformer-based spectral modeling for hyperspectral classification.
- [SSFTT](https://ieeexplore.ieee.org/document/9717899): spectral-spatial feature tokenization transformer.
- [Linformer](https://arxiv.org/abs/2006.04768), [Performer](https://arxiv.org/abs/2009.14794), [Nyströmformer](https://arxiv.org/abs/2102.03902), [FlashAttention](https://arxiv.org/abs/2205.14135): efficient attention references.
