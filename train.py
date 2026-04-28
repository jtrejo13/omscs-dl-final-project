"""
train.py
--------
NAFNet-style training script for local execution.

Usage
-----
    python train.py --opt experiments/train.yml
    python train.py --opt experiments/train.yml --resume results/<name>/checkpoints/latest.pth

The script will:
  1. Parse options from the YAML file.
  2. Set the global RNG seed for reproducibility.
  3. Build train + val dataloaders (train loader gets a seeded generator).
  4. Resolve scheduler.T_max against epochs * iters_per_epoch.
  5. Instantiate the model.
  6. Optionally resume from a checkpoint.
  7. Run the training loop with NaN-loss guard.
  8. Evaluate PSNR / SSIM on the val set at configurable intervals.
  9. Save latest.pth every epoch + epoch_NNNN.pth at save_freq.
"""

import argparse
import hashlib
import os
import pathlib
import random
import time

import numpy as np
import torch

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


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init: re-seed numpy/random per worker"""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _resolve_t_max(opt: dict, iters_per_epoch: int) -> None:
    """Auto-compute scheduler.T_max if missing; abort if a hardcoded value
    disagrees with epochs * iters_per_epoch"""
    train_cfg = opt.setdefault("train", {})
    sched_cfg = train_cfg.setdefault("scheduler", {})
    epochs = int(train_cfg.get("epochs", 50))
    expected = iters_per_epoch * epochs

    if "T_max" in sched_cfg:
        configured = int(sched_cfg["T_max"])
        if configured != expected:
            raise ValueError(
                f"\nscheduler.T_max mismatch.\n"
            )
        print(f"Validated T_max = {configured}")
    else:
        sched_cfg["T_max"] = expected
        print(
            f"Auto-set T_max = {expected} "
            f"({epochs} epochs * {iters_per_epoch} iters/epoch)"
        )


def _wandb_init(opt: dict) -> bool:
    if not _WANDB_AVAILABLE:
        return False
    has_key = bool(os.environ.get("WANDB_API_KEY"))
    netrc = pathlib.Path.home() / ".netrc"
    has_netrc = netrc.exists() and "wandb" in netrc.read_text()
    if not (has_key or has_netrc):
        print("[W&B] No credentials; run `wandb login` or set WANDB_API_KEY. Skipping.")
        return False

    exp_name = opt.get("name", "experiment")
    run_id = hashlib.md5(exp_name.encode()).hexdigest()[:16]

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "nafnet-sidd"),
        name=exp_name,
        id=run_id,
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a training checkpoint to resume from. "
            "Overrides opt['path']['resume'] in the YAML if both are set."
        ),
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


def train(opt: dict, resume_override: str | None = None):
    exp_name = opt.get("name", "experiment")
    ckpt_dir = os.path.join("results", exp_name, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_cfg = opt.get("train", {})
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    print(f"[Seed] global seed = {seed}")

    torch.backends.cudnn.benchmark = True

    use_wandb = _wandb_init(opt)

    train_gen = torch.Generator()
    train_gen.manual_seed(seed)
    train_loader = build_dataloader(
        opt, phase="train",
        worker_init_fn=seed_worker, generator=train_gen,
    )
    val_loader = build_dataloader(opt, phase="val")
    print(
        f"[Data] train batches={len(train_loader)}  " f"val batches={len(val_loader)}"
    )

    _resolve_t_max(opt, len(train_loader))

    model = build_model(opt)
    print(f"[Model] type = {opt.get('model', {}).get('type', 'dummy')}")

    total_epochs = int(train_cfg.get("epochs", 50))
    log_freq = int(train_cfg.get("log_freq", 10))  # iters
    val_freq = int(train_cfg.get("val_freq", 5))  # epochs
    save_freq = int(train_cfg.get("save_freq", 10))  # epochs
    nan_abort_threshold = int(train_cfg.get("nan_abort_threshold", 50))

    resume_path = resume_override or opt.get("path", {}).get("resume")
    start_epoch = 1
    global_iter = 0
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = model.load(resume_path)
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
            global_iter = int(ckpt.get("global_iter", 0))
            print(
                f"[Resume] from epoch {ckpt['epoch']}, iter {global_iter} "
                f"-> continuing at epoch {start_epoch}"
            )
            if start_epoch > total_epochs:
                print(
                    f"[Resume] checkpoint already at/past total_epochs={total_epochs}; "
                    f"nothing to do."
                )
                if use_wandb:
                    wandb.finish()
                return
        else:
            print(
                "[Resume] checkpoint has no train state (likely pretrained-only); "
                "starting fresh from epoch 1 with loaded weights."
            )

    print(f"\n{'='*60}")
    print(f"  Starting training: {exp_name}")
    print(
        f"  Epochs : {start_epoch} -> {total_epochs}  |  "
        f"Val every {val_freq} epochs  |  "
        f"Save every {save_freq} epochs"
    )
    print(f"{'='*60}\n")

    loss_meter = AverageMeter("Loss")
    nan_streak = 0

    for epoch in range(start_epoch, total_epochs + 1):
        loss_meter.reset()
        epoch_start = time.time()

        for _, batch in enumerate(train_loader):
            model.feed_data(batch)
            logs = model.optimize()
            global_iter += 1

            if logs.get("skipped", False):
                nan_streak += 1
                if nan_streak >= nan_abort_threshold:
                    raise RuntimeError(
                        f"NaN/Inf loss for {nan_streak} consecutive iterations "
                        f"(iter {global_iter}); aborting to avoid corrupting the run. "
                        f"Inspect data integrity, gradient clip, and loss epsilon."
                    )
                if nan_streak <= 5 or nan_streak % 10 == 0:
                    print(
                        f"  [WARN] NaN/Inf loss at iter {global_iter} "
                        f"(streak={nan_streak}); skipping batch."
                    )
                if use_wandb:
                    wandb.log({"train/nan_streak": nan_streak}, step=global_iter)
                continue

            if nan_streak > 0:
                print(f"  [INFO] NaN streak ended at length {nan_streak}.")
                nan_streak = 0

            loss_meter.update(logs["loss"])

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

        latest_path = os.path.join(ckpt_dir, "latest.pth")
        model.save(latest_path, epoch=epoch, global_iter=global_iter)

        # Periodic snapshots for retrospective analysis.
        if epoch % save_freq == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth")
            model.save(ckpt_path, epoch=epoch, global_iter=global_iter)

    if use_wandb:
        wandb.finish()
    print("\nTraining complete.")


if __name__ == "__main__":
    args = parse_args()
    opt = load_config(args.opt)
    train(opt, resume_override=args.resume)
