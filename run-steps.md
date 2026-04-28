# Run Steps

## 1. Provision the Vast.ai Instance

1. Go to [vast.ai](https://vast.ai) and search for an instance with at least 24 GB VRAM (RTX 3090 / 4090) and 100 GB disk. Look for "DLPerf" and especially "Inet Down" ≥ 500 Mbps (ideally 1 Gbps+)
2. Select a PyTorch template (e.g. `pytorch/pytorch:2.1.0-cuda12.1...); this avoids reinstalling CUDA.
3. Start the instance and wait for it to show **Running**.
4. Click **Connect** --> copy the SSH command and run it in your terminal.

## 2. Clone the Repo

```bash
git clone https://github.com/jtrejo13/omscs-dl-final-project omscs-dl-final-project
cd omscs-dl-final-project
git checkout <your-branch>  (if running your own code)
```

## 3. Log in to HuggingFace

Do this before running `setup.sh`; the setup script downloads ~70 GB of SIDD LMDBs from Huggingface and will fail without credentials.

```bash
pip install huggingface_hub hf_transfer
hf auth login
```

Token is at [huggingface.co --> Settings --> Access Tokens](https://huggingface.co/settings/tokens).

## 4. Log in to W&B

```bash
pip install wandb
wandb login
```

Key is at [wandb.ai --> Settings --> API Keys](https://wandb.ai/settings).

## 5. Run Setup

```bash
bash setup.sh
```

This will:
- Create the `phoenix` conda environment from `environment.yml`
- Download the SIDD train + val LMDBs (~70 GB) from `cdtrejo/sidd-medium-lmdb` on Huggingface
- Run a 1-epoch smoke test to validate the full pipeline

## 6. Activate the Environment

```bash
conda activate phoenix
```

Verify GPU access:

```bash
nvidia-smi && python -c "import torch; assert torch.cuda.is_available()"
```

## 7. Run Training

```bash
# Baseline
python train.py --opt experiments/train_baseline.yml

# Variant A (in a separate session or after baseline finishes)
python train.py --opt experiments/train_variant_<a|b|c>.yml
```

To keep training alive after disconnecting from SSH, use `tmux` (recommended) or `nohup`:

```bash
# Option A: tmux (lets you reattach)
tmux new -s baseline
python train.py --opt experiments/train_baseline.yml
# Ctrl+B then D to detach; tmux attach -t baseline to reattach

# Option B: nohup
mkdir -p logs
nohup python train.py --opt experiments/train_baseline.yml > logs/baseline.out 2>&1 &
```

If the instance is preempted, resume from the latest checkpoint:

```bash
python train.py --opt experiments/train_baseline.yml --resume results/nafnet_sidd_baseline/checkpoints/latest.pth
```

## 8. Evaluate and Upload to HuggingFace

```bash
python test.py --opt experiments/test_baseline.yml --save_images
python test.py --opt experiments/test_variant_<a|b|c>.yml --save_images
```

This evaluates PSNR/SSIM/LPIPS, saves `results.json`, and automatically uploads `latest.pth` + `results.json` to `cdtrejo/nafnet-sidd-checkpoints` on Huggingface.
