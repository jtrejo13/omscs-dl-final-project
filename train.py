"""
train.py
--------
NAFNet-style training script for local execution.

Usage
-----
    python train.py --opt experiments/train.yml

The script will:
  1. Parse options from the YAML file.
  2. Build train + val dataloaders.
  3. Instantiate the model (defaults to DummyRestorationModel).
  4. Run the training loop, logging loss each iteration.
  5. Evaluate PSNR / SSIM on the val set at configurable intervals.
  6. Save checkpoints to results/<name>/checkpoints/.
"""

import argparse
import os
import pathlib
import time

from data import build_dataloader
from models import build_model
from utils import load_config, compute_psnr, compute_ssim, AverageMeter

_WANDB_DISABLED = os.environ.get("WANDB_DISABLED", "false").lower() in ("1", "true", "yes")
try:
    if _WANDB_DISABLED:
        raise ImportError
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


def _wandb_init(opt: dict) -> bool:
    if not _WANDB_AVAILABLE:
        return False
    has_key = bool(os.environ.get("WANDB_API_KEY"))
    netrc = pathlib.Path.home() / ".netrc"
    has_netrc = netrc.exists() and "wandb" in netrc.read_text()
    if not (has_key or has_netrc):
        print("[W&B] No credentials; run `wandb login` or set WANDB_API_KEY. Skipping.")
        return False
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "nafnet-sidd"),
        name=opt.get("name", "experiment"),
        config={"model": opt.get("model", {}), "train": opt.get("train", {})},
        resume="allow",
    )
    print(f"[W&B] {wandb.run.url}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--opt",
        type=str,
        default="experiments/train_dummy.yml",
        help="Path to the YAML options file",
    )
    return parser.parse_args()


def validate(model, val_loader) -> dict:
    """Run model on val set and return avg PSNR and SSIM"""
    psnr_meter = AverageMeter("PSNR")
    ssim_meter = AverageMeter("SSIM")

    for batch in val_loader:
        model.feed_data(batch)
        model.test()
        visuals = model.get_current_visuals()
        pred = visuals["pred"]
        gt = visuals["gt"]
        psnr_meter.update(compute_psnr(pred, gt), n=pred.shape[0])
        ssim_meter.update(compute_ssim(pred, gt), n=pred.shape[0])

    return {"psnr": psnr_meter.avg, "ssim": ssim_meter.avg}


def train(opt: dict):
    exp_name = opt.get("name", "experiment")
    ckpt_dir = os.path.join("results", exp_name, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    use_wandb = _wandb_init(opt)

    train_loader = build_dataloader(opt, phase="train")
    val_loader = build_dataloader(opt, phase="val")
    print(
        f"[Data] train batches={len(train_loader)}  " f"val batches={len(val_loader)}"
    )

    model = build_model(opt)
    print(f"[Model] type = {opt.get('model', {}).get('type', 'dummy')}")

    train_cfg = opt.get("train", {})
    total_epochs = int(train_cfg.get("epochs", 50))
    log_freq = int(train_cfg.get("log_freq", 10))  # iters
    val_freq = int(train_cfg.get("val_freq", 5))  # epochs
    save_freq = int(train_cfg.get("save_freq", 10))  # epochs

    print(f"\n{'='*60}")
    print(f"  Starting training: {exp_name}")
    print(
        f"  Epochs : {total_epochs}  |  "
        f"Val every {val_freq} epochs  |  "
        f"Save every {save_freq} epochs"
    )
    print(f"{'='*60}\n")

    global_iter = 0
    loss_meter = AverageMeter("Loss")

    for epoch in range(1, total_epochs + 1):
        loss_meter.reset()
        epoch_start = time.time()

        for _, batch in enumerate(train_loader):
            model.feed_data(batch)
            logs = model.optimize()
            loss_meter.update(logs["loss"])
            global_iter += 1

            if global_iter % log_freq == 0:
                print(
                    f"  [Epoch {epoch:>3}/{total_epochs}  "
                    f"iter {global_iter:>6}]  "
                    f"loss={loss_meter.avg:.5f}"
                )
                if use_wandb:
                    wandb.log({"train/loss": loss_meter.avg}, step=global_iter)

        epoch_time = time.time() - epoch_start
        print(
            f"  Epoch {epoch:>3} done  "
            f"avg_loss={loss_meter.avg:.5f}  "
            f"time={epoch_time:.1f}s"
        )
        if use_wandb:
            wandb.log({"epoch/loss": loss_meter.avg, "epoch/time_s": epoch_time, "epoch": epoch}, step=global_iter)

        if epoch % val_freq == 0:
            val_metrics = validate(model, val_loader)
            print(
                f"  >>> Val   PSNR={val_metrics['psnr']:.2f} dB  "
                f"SSIM={val_metrics['ssim']:.4f}"
            )
            if use_wandb:
                wandb.log({"val/psnr": val_metrics["psnr"], "val/ssim": val_metrics["ssim"], "epoch": epoch}, step=global_iter)

        if epoch % save_freq == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth")
            model.save(ckpt_path)

    model.save(os.path.join(ckpt_dir, "latest.pth"))
    if use_wandb:
        wandb.finish()
    print("\nTraining complete.")


if __name__ == "__main__":
    args = parse_args()
    opt = load_config(args.opt)
    train(opt)
