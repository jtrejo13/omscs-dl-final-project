"""
test.py
-------
NAFNet-style evaluation / inference script for local execution.

Usage
-----
    # Evaluate with PSNR & SSIM (requires ground truth images):
    python test.py --opt experiments/test_dummy.yml

The script will:
  1. Parse options from the YAML file.
  2. Build the test dataloader.
  3. Instantiate and load the model from a checkpoint.
  4. Run inference on every sample.
  5. Print per-image PSNR / SSIM and aggregate averages.
  6. Optionally save output images to results/<name>/images/.
"""

import argparse
import datetime
import json
import os
import pathlib

from torchvision.utils import save_image

from data import build_dataloader
from models import build_model
from utils import (
    load_config,
    compute_psnr,
    compute_ssim,
    compute_lpips,
    build_lpips_fn,
    AverageMeter,
    get_device,
)

HF_MODEL_REPO = "cdtrejo/nafnet-sidd-checkpoints"

_WANDB_DISABLED = os.environ.get("WANDB_DISABLED", "false").lower() in ("1", "true", "yes")
try:
    if _WANDB_DISABLED:
        raise ImportError
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


def _wandb_active() -> bool:
    if not _WANDB_AVAILABLE:
        return False
    has_key = bool(os.environ.get("WANDB_API_KEY"))
    netrc = pathlib.Path.home() / ".netrc"
    has_netrc = netrc.exists() and "wandb" in netrc.read_text()
    return has_key or has_netrc


def _upload_to_hf(exp_name: str, ckpt_path: str, results_path: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[HF] huggingface_hub not installed; skipping upload.")
        return

    api = HfApi()
    try:
        print(f"[HF] Uploading to {HF_MODEL_REPO}/{exp_name}/...")
        api.upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo=f"{exp_name}/latest.pth",
            repo_id=HF_MODEL_REPO,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj=results_path,
            path_in_repo=f"{exp_name}/results.json",
            repo_id=HF_MODEL_REPO,
            repo_type="model",
        )
        print(f"[HF] Upload complete: huggingface.co/{HF_MODEL_REPO}")
    except Exception as e:
        print(f"[HF] Upload failed: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test / evaluation script")
    parser.add_argument(
        "--opt",
        type=str,
        default="experiments/test_dummy.yml",
        help="Path to the YAML options file",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save output images to results/<name>/images/",
    )
    return parser.parse_args()


def test(opt: dict, save_images: bool = False):
    exp_name = opt.get("name", "experiment")
    result_dir = os.path.join("results", exp_name, "images")
    if save_images:
        os.makedirs(result_dir, exist_ok=True)

    test_loader = build_dataloader(opt, phase="test")
    print(f"[Data] test batches = {len(test_loader)}")

    model = build_model(opt)

    ckpt_path = opt.get("path", {}).get("pretrain_model", None)
    if ckpt_path:
        model.load(ckpt_path)
    else:
        print("[Warning] No checkpoint path specified; using random weights.")

    psnr_meter = AverageMeter("PSNR")
    ssim_meter = AverageMeter("SSIM")

    lpips_fn = build_lpips_fn(net="alex", device=get_device())
    lpips_meter = AverageMeter("LPIPS")
    if lpips_fn is None:
        print("[Warning] lpips not installed; LPIPS will be skipped.")
    per_sample_results = []

    print(f"\n{'='*60}")
    print(f"  Testing: {exp_name}")
    print(f"{'='*60}\n")

    for idx, batch in enumerate(test_loader):
        model.feed_data(batch)
        model.test()
        visuals = model.get_current_visuals()

        pred = visuals["pred"]  # (B, C, H, W)  in [0, 1]
        paths = batch.get("path", [f"sample_{idx}"])

        if "gt" in visuals:
            gt = visuals["gt"]
            psnr = compute_psnr(pred, gt)
            ssim = compute_ssim(pred, gt)
            lpips_ = compute_lpips(pred, gt, lpips_fn)

            psnr_meter.update(psnr, n=pred.shape[0])
            ssim_meter.update(ssim, n=pred.shape[0])
            if lpips_ is not None:
                lpips_meter.update(lpips_, n=pred.shape[0])

            sample_name = os.path.splitext(os.path.basename(str(paths[0])))[0]
            lpips_str = f"  LPIPS={lpips_:.4f}" if lpips_ is not None else ""
            print(
                f"  [{idx+1:>4}/{len(test_loader)}]  "
                f"{os.path.basename(str(paths[0])): <30}  "
                f"PSNR={psnr:.2f} dB  SSIM={ssim:.4f}{lpips_str}"
            )

            per_sample_results.append(
                {
                    "sample": sample_name,
                    "psnr": round(psnr, 6) if psnr != float("inf") else None,
                    "ssim": round(ssim, 6),
                    "lpips": round(lpips_, 6) if lpips_ is not None else None,
                }
            )

        if save_images:
            for b in range(pred.shape[0]):
                fname = (
                    os.path.basename(str(paths[b]))
                    if b < len(paths)
                    else f"sample_{idx}_{b}.png"
                )

                fname = os.path.splitext(fname)[0] + ".png"
                save_image(pred[b], os.path.join(result_dir, fname))

    if psnr_meter.count > 0:
        avg_lpips = lpips_meter.avg if lpips_meter.count > 0 else None
        print(f"\n  ── Average Results ──────────────────────────────")
        print(f"  PSNR : {psnr_meter.avg:.4f} dB")
        print(f"  SSIM : {ssim_meter.avg:.4f}")
        if avg_lpips is not None:
            print(f"  LPIPS: {avg_lpips:.4f}")
        print(f"  ─────────────────────────────────────────────────\n")

        results_path = os.path.join("results", exp_name, "results.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        payload = {
            "experiment": exp_name,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "metrics": {
                "psnr_avg": round(psnr_meter.avg, 4),
                "ssim_avg": round(ssim_meter.avg, 4),
                "lpips_avg": round(avg_lpips, 4) if avg_lpips is not None else None,
                "n_samples": psnr_meter.count,
            },
            "per_sample": per_sample_results,
        }
        with open(results_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Saved results: {results_path}")

        # Log final metrics to W&B
        if _wandb_active():
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "nafnet-sidd"),
                name=exp_name,
                resume="allow",
            )
            wandb.summary["test/psnr"] = payload["metrics"]["psnr_avg"]
            wandb.summary["test/ssim"] = payload["metrics"]["ssim_avg"]
            if payload["metrics"]["lpips_avg"] is not None:
                wandb.summary["test/lpips"] = payload["metrics"]["lpips_avg"]
            wandb.finish()
            print("[W&B] Test metrics logged.")

        # Upload checkpoint + results to HuggingFace
        ckpt_path = os.path.join("results", exp_name, "checkpoints", "latest.pth")
        if os.path.exists(ckpt_path):
            _upload_to_hf(exp_name, ckpt_path, results_path)

    if save_images:
        print(f"  Saved outputs: {result_dir}\n")

    print("Testing complete.")


if __name__ == "__main__":
    args = parse_args()
    opt = load_config(args.opt)
    test(opt, save_images=args.save_images)
