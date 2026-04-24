"""
utils/metrics.py
----------------
PSNR and SSIM implementations that work on CPU/GPU torch tensors.
Both functions expect tensors in (B, C, H, W) or (C, H, W) shape,
with pixel values in [0, 1].
"""

import math
import lpips
import torch
import torch.nn.functional as F


def get_device() -> str:
    """Auto-detect the device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (dB), averaged over the batch."""
    pred = pred.detach().clamp(0.0, max_val)
    gt = gt.detach().clamp(0.0, max_val)
    mse = F.mse_loss(pred, gt, reduction="mean").item()
    if mse < 1e-10:
        return float("inf")
    return 10.0 * math.log10(max_val**2 / mse)


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    kernel = g.unsqueeze(0) * g.unsqueeze(1)  # (W, W)
    return kernel


def compute_ssim(
    pred: torch.Tensor,
    gt: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    c1: float = 0.01**2,
    c2: float = 0.03**2,
) -> float:
    """Structural Similarity Index (averaged over batch and channels)"""
    pred = pred.detach().clamp(0.0, 1.0)
    gt = gt.detach().clamp(0.0, 1.0)

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)

    B, C, H, W = pred.shape
    device = pred.device

    kernel = _gaussian_kernel(window_size, sigma).to(device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, W, W)
    kernel = kernel.expand(C, 1, window_size, window_size)
    pad = window_size // 2

    def conv(x):
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu_x = conv(pred)
    mu_y = conv(gt)
    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y

    sigma_x_sq = conv(pred * pred) - mu_x_sq
    sigma_y_sq = conv(gt * gt) - mu_y_sq
    sigma_xy = conv(pred * gt) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )

    return ssim_map.mean().item()


def build_lpips_fn(net: str = "alex", device: str = "cpu"):
    """Create an lpips.LPIPS model to pass to compute_lpips"""
    loss_fn = lpips.LPIPS(net=net, verbose=False).eval()
    try:
        loss_fn.to(device)
    except Exception:
        loss_fn.to("cpu")
    return loss_fn


def compute_lpips(pred: torch.Tensor, gt: torch.Tensor, loss_fn) -> float | None:
    """LPIPS score (lower = better), averaged over batch. Expects [0, 1] tensors"""
    if loss_fn is None:
        return None
    pred = pred.detach().clamp(0.0, 1.0)
    gt = gt.detach().clamp(0.0, 1.0)
    if pred.dim() == 3:
        pred, gt = pred.unsqueeze(0), gt.unsqueeze(0)
    device = next(loss_fn.parameters()).device
    with torch.no_grad():
        p = (pred * 2.0 - 1.0).to(device)
        g = (gt * 2.0 - 1.0).to(device)
        score = loss_fn(p, g)
    return score.mean().item()


class AverageMeter:
    """Tracks a running mean"""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"
