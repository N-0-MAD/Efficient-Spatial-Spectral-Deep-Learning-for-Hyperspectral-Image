# Detailed Research Pipeline

## 1. Problem Definition

Independent band tokenization treats each spectral band as its own token stream. For a patch grid with `P` spatial patches and `C` bands, the token count becomes:

```text
tokens = P * C
```

Self-attention then has a rough cost proxy of:

```text
attention_cost = tokens^2
```

This project asks whether we can keep useful spectral reasoning while avoiding full attention over all band-patch tokens.

## 2. Dataset Strategy

Start with Sentinel-2 L2A because it is easy to access through STAC and has stable multispectral bands. Use it to validate the pipeline.

Then add a hyperspectral dataset with more than 50 bands to stress-test scalability. Candidate families include EnMAP, EMIT, AVIRIS, PRISMA, DESIS, or public benchmark cubes such as Indian Pines, Pavia University, Houston, or Chikusei.

## 3. Data Format

Every scene should become a tensor:

```text
(C, H, W)
```

Where:

- `C` is the spectral dimension.
- `H, W` are spatial dimensions.

Every patch should become:

```text
(C, patch_size, patch_size)
```

## 4. Preprocessing

Recommended preprocessing order:

1. Load reflectance bands.
2. Apply cloud and no-data masking where available.
3. Resample bands to a common ground sampling distance.
4. Normalize reflectance values.
5. Crop image edges to exact patch multiples.
6. Extract fixed-size patches.
7. Attach labels and metadata.

## 5. Tokenization Experiments

Compare:

- independent band tokenization,
- all-band patch tokenization,
- grouped-band tokenization,
- larger spatial patches,
- dual-resolution tokenization.

The goal is to show how token count and attention cost change as band count grows.

## 6. Baselines

Minimum baseline set:

1. CNN-only baseline.
2. Full transformer with independent band tokens.
3. Simple hybrid CNN + spectral attention model with concatenation fusion.

These give fair anchors for accuracy and compute.

## 7. Proposed Architecture

```text
Input: multi-band patch (C, H, W)

Spatial branch:
  CNN/local operators over H, W

Spectral branch:
  attention over bands or band groups

Fusion:
  concat -> gated fusion -> cross-attention

Output:
  classification head for V1
```

## 8. Evaluation

Report:

- accuracy,
- macro F1,
- class-wise F1,
- parameter count,
- FLOPs,
- peak GPU memory,
- inference time,
- token count,
- attention-cost proxy,
- scaling curves as band count increases.

## 9. Ablations

Run:

- CNN-only vs transformer-only vs hybrid,
- concat vs gated vs cross-attention fusion,
- independent band tokens vs grouped tokens,
- small vs large patches,
- Sentinel-2 bands vs hyperspectral bands,
- full attention vs spectral-only attention.

## 10. Success Criterion

The project is successful if the hybrid model reaches near-transformer accuracy with lower memory/runtime, especially as band count increases.

