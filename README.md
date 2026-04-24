# NAFNet Variants for Image Denoising

> Deep Learning project that compares architectural variants of NAFNet (a CNN) on the task of image denoising.

**Team: Phoenix** — Juan Trejo, Carlos Martinez, Amirhossein Khalighi
**Course:** CS 7643 OMSCS Deep Learning



## Quickstart

**conda (recommended):**
```bash
conda env create -f environment.yml
conda activate phoenix
```

**pip:**
```bash
pip install -r requirements.txt
```

```bash
# Train
python train.py --opt experiments/train_dummy.yml

# Evaluate
python test.py --opt experiments/test_dummy.yml
```

## Overview

High-quality image restoration — recovering clean images from degraded or noisy inputs — is an important problem in archival preservation, medical imaging, and consumer media. This project studies **NAFNet**, a purely CNN-based architecture that achieves state-of-the-art restoration quality by eliminating non-linear activation functions in favor of a learned gating mechanism (SimpleGate) and a lightweight channel attention module (SCA).

We modify NAFNet's core design choices by training the baseline alongside three targeted variants, each modifying exactly one component, to understand which parts of the architecture actually drive performance.

## Goals

- Establish a **NAFNet baseline**.
- Train and evaluate **three architectural variants**, each isolating one design decision.
- Measure the effect of each modification on restoration quality and computational efficiency.
- Write a final report analyzing what the ablations reveal about NAFNet's design.

## Models

| Model | Modification | What it tests |
|---|---|---|
| **BaselineNAFNet** | None | Reference point |
| **Variant A** — GELU Gate | `x1 * GELU(x2)` instead of `x1 * x2` | Role of the activation-free gate |
| **Variant B** — No SCA | SCA module removed | Contribution of channel attention |
| **Variant C** — BatchNorm | LayerNorm → BatchNorm2d | Normalization strategy |

## Dataset

**SIDD**: https://abdokamel.github.io/sidd/

## Evaluation

**Quality metrics**
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

**Efficiency metrics**
- FLOPs
- Parameter count
- Inference time
- Peak VRAM
