# NAFNet Variants for Image Denoising

> Deep Learning project that compares architectural variants of NAFNet (a CNN) on the task of image denoising.

**Team: Phoenix**: Juan Trejo, Carlos Martinez, Amirhossein Khalighi
**Course:** CS 7643 OMSCS Deep Learning

## Overview

High-quality image restoration, recovering clean images from degraded or noisy inputs, is an important problem
in archival preservation, medical imaging, and consumer media. This project studies **NAFNet**, a purely
CNN-based architecture that achieves state-of-the-art restoration quality by eliminating non-linear activation
functions in favor of a learned gating mechanism (SimpleGate) and a lightweight channel attention module (SCA).

We modify NAFNet's core design choices by training the baseline alongside three targeted variants,
each modifying exactly one component, to understand which parts of the architecture actually drive performance.

## Goals

- Establish a **NAFNet baseline**.
- Train and evaluate **three architectural variants**, each isolating one design decision.
- Measure the effect of each modification on restoration quality and computational efficiency.
- Write a final report analyzing what the ablations reveal about NAFNet's design.

## Models

| Model | Modification | What it tests |
|---|---|---|
| **BaselineNAFNet** | None | Reference point |
| **Variant A**: GELU Gate | `x1 * GELU(x2)` instead of `x1 * x2` | Role of the activation-free gate |
| **Variant B**: No SCA | SCA module removed | Contribution of channel attention |
| **Variant C**: BatchNorm | LayerNorm → BatchNorm2d | Normalization strategy |

## Dataset

**SIDD Medium**: https://abdokamel.github.io/sidd/

Pre-processed LMDBs are stored on PACE at:
```
/storage/ice-shared/cs7643/shared-group-project-data/data/SIDD/
├── train/
│   ├── input_crops.lmdb   # 30,608 noisy patches (512×512)
│   └── gt_crops.lmdb      # 30,608 clean patches (512×512)
└── val/
    ├── input_crops.lmdb   # 1,280 noisy patches (256×256)
    └── gt_crops.lmdb      # 1,280 clean patches (256×256)
```

## Evaluation Metrics

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

## Running on PACE

### 1. Clone and set up the environment

```bash
cd /storage/ice-shared/cs7643/shared-group-project-data/phoenix
git clone <repo-url>
cd omscs-dl-final-project

conda env create -f environment.yml
conda activate phoenix
```

To update an existing environment after dependency changes:
```bash
conda env update -n phoenix -f environment.yml --prune
```

### 2. Optional Smoke test (validates the full pipeline before committing to a full run)

```bash
sbatch jobs/train_smoke.sh
```

This trains the baseline for 1 epoch on 80 samples and immediately runs evaluation. Check the output:
```bash
tail -f logs/smoke_<jobid>.out
```

A successful smoke test prints training loss, then PSNR / SSIM / LPIPS on the val set.

### 3. Run a full training job

Each model has a dedicated train config and batch script:

| Model | Config | Job |
|---|---|---|
| Baseline | `experiments/train_baseline.yml` | `jobs/train_baseline.sh` |
| Variant A | `experiments/train_variant_a.yml` | `jobs/train_variant_a.sh` |
| Variant B | `experiments/train_variant_b.yml` | `jobs/train_variant_b.sh` |
| Variant C | `experiments/train_variant_c.yml` | `jobs/train_variant_c.sh` |

```bash
sbatch jobs/train_baseline.sh
```

### 4. Evaluate a trained model

```bash
sbatch jobs/test_baseline.sh
```

Results are written to `results/<name>/results.json` (PSNR, SSIM, LPIPS per image + averages).

## Adding a New Experiment

### 1. Implement the model

Add `models/variant_x.py` following the same interface as `models/baseline_nafnet.py`:

```python
class VariantX(BaselineNAFNet):
    def __init__(self, opt):
        super().__init__(opt)
        # TODO: modify self.net here
```

Register it in `models/__init__.py`:
```python
elif model_type == "variant_x":
    from models.variant_x import VariantX
    return VariantX(opt)
```

### 2. Create experiment configs

Copy an existing config pair and update `name` and `model.type`:

```bash
cp experiments/train_baseline.yml experiments/train_variant_x.yml
cp experiments/test_baseline.yml  experiments/test_variant_x.yml
```

Edit both files and change these two fields:
```yaml
name: nafnet_sidd_variant_x
model:
  type: variant_x
```

Also update `path.pretrain_model` in the test config:
```yaml
path:
  pretrain_model: results/nafnet_sidd_variant_x/checkpoints/latest.pth
```

### 3. Create batch job scripts

```bash
cp jobs/train_baseline.sh jobs/train_variant_x.sh
cp jobs/test_baseline.sh  jobs/test_variant_x.sh
```

Update the `--job-name` and the config path in each:
```bash
#SBATCH --job-name=nafnet_variant_x
python train.py --opt experiments/train_variant_x.yml
```

### 4. Verify locally before submitting

```bash
python -c "
from utils.options import load_config
from models import build_model
opt = load_config('experiments/train_variant_x.yml')
m = build_model(opt)
print(f'ok; {sum(p.numel() for p in m.net.parameters()):,} params')
"
```

### 5. Submit

```bash
sbatch jobs/train_variant_x.sh
```

## Running on Vast.ai

End-to-end guide for running a variant on a rented GPU instance.

### Accounts needed

| Service | Purpose | Notes |
|---|---|---|
| [vast.ai](https://vast.ai) | Rented GPU | Create a free account, add $5–$10 credit |
| [wandb.ai](https://wandb.ai) | Metric tracking | Get team API key |
| [huggingface.co](https://huggingface.co) | Dataset + checkpoint storage | Use HF token |

### 1. Rent an instance

1. Go to [vast.ai/create](https://vast.ai/create)
2. Set filters: **GPU = RTX 3090**, **GPU RAM ≥ 20 GB**, **Disk = 150 GB**
3. Docker image can be `vastai/pytorch` with a **CUDA 12.1+** tag
4. Expand **Environment Variables** and add:
   ```
   WANDB_API_KEY=<your key from wandb.ai/settings>
   HF_TOKEN=<token from huggingface.co/settings/tokens with write access>
   ```
   We still need to confirm this.
5. Click **Rent** and wait for status to show **Running**
6. Click **Connect** and copy the SSH command, then connect:
   ```bash
   ssh -p <port> root@<ip>
   ```

### 2. Set up the instance

```bash
git clone -b <my-branch> https://github.com/jtrejo13/omscs-dl-final-project omscs-dl-final-project
cd omscs-dl-final-project
bash setup.sh
```

### 3. Run variant

For **Variant A**:
```bash
python train.py --opt experiments/train_variant_a.yml
python test.py  --opt experiments/test_variant_a.yml
```

### 4. Save results

`test.py` automatically uploads `latest.pth` and `results.json` to HuggingFace when it finishes. If it fails, run manually:

```bash
# Replace nafnet_sidd_variant_b with nafnet_sidd_variant_c as needed
hf upload cdtrejo/nafnet-sidd-checkpoints results/nafnet_sidd_variant_b/checkpoints/latest.pth nafnet_sidd_variant_b/latest.pth --repo-type model

hf upload cdtrejo/nafnet-sidd-checkpoints results/nafnet_sidd_variant_b/results.json nafnet_sidd_variant_b/results.json --repo-type model
```

Results for all models will appear at `huggingface.co/cdtrejo/nafnet-sidd-checkpoints`.

### 5. Destroy the instance

Once results are uploaded, destroy the instance from the Vast.ai dashboard to stop billing.

## Local Development

```bash
conda env create -f environment.yml
conda activate phoenix

# Quick smoke test (uses synthetic data; no SIDD required)
python train.py --opt experiments/train_smoke.yml
python test.py  --opt experiments/test_smoke.yml
```
