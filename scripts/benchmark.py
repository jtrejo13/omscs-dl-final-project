"""
scripts/benchmark.py
--------------------
Compute efficiency metrics for NAFNet ablation variants.

Metrics measured:
  1. FLOPs / MACs   — via thop.profile() on a fixed 256×256 input
  2. Inference latency (ms/image) — 100 warm-up + 500 timed runs
  3. Peak GPU memory (MB)  — torch.cuda.max_memory_allocated()
  4. Model size on disk (MB) — os.path.getsize(checkpoint)
  5. PSNR per GFLOP — derived from existing results.json

Usage:
    # Benchmark all 6 canonical variants:
    python scripts/benchmark.py

    # Benchmark a single variant:
    python scripts/benchmark.py --opt experiments/test_baseline.yml

    # Custom input resolution:
    python scripts/benchmark.py --input_size 512

    # Custom number of timed runs:
    python scripts/benchmark.py --warmup 50 --runs 200
"""

import argparse
import json
import os
import sys
import time

import torch
import yaml

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so we can import models/utils
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models import build_model  # noqa: E402
from utils import load_config    # noqa: E402

# ---------------------------------------------------------------------------
# Canonical variant configs — curated list to avoid duplicates (_ak, etc.)
# ---------------------------------------------------------------------------
CANONICAL_CONFIGS = [
    "experiments/test_baseline.yml",
    "experiments/test_variant_a.yml",
    "experiments/test_variant_b_ctrj.yml",
    "experiments/test_variant_c_ctrj.yml",
    "experiments/test_variant_e.yml",
    "experiments/test_variant_f.yml",
]

# Display names for the paper table
DISPLAY_NAMES = {
    "nafnet_sidd_baseline": "Baseline",
    "nafnet_sidd_variant_a": "A (GELU gate)",
    "nafnet_sidd_variant_b_ctrj": "B (no SCA)",
    "nafnet_sidd_variant_c_ctrj": "C (BatchNorm)",
    "nafnet_sidd_variant_c_ctrj_b16": "C (BatchNorm)",
    "nafnet_sidd_variant_e": "E (asym. gate)",
    "nafnet_sidd_variant_f": "F (gated skip)",
}


def get_device() -> str:
    """Auto-detect device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_parameters(model: torch.nn.Module) -> int:
    """Total number of learnable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops(model: torch.nn.Module, input_size: int, device: str):
    """Measure FLOPs/MACs using thop.profile().

    Returns (macs, params) where macs is the number of multiply-accumulate
    operations. FLOPs ≈ 2 × MACs for most operations.
    """
    try:
        from thop import profile, clever_format
    except ImportError:
        print("[Warning] thop not installed; skipping FLOPs measurement.")
        print("         Install with: pip install thop")
        return None, None

    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(dummy,), verbose=False)
    return macs, params


def measure_latency(model: torch.nn.Module, input_size: int, device: str,
                    warmup: int = 100, runs: int = 500):
    """Measure inference latency in ms/image.

    Uses torch.cuda.synchronize() on CUDA for accurate timing.
    Returns (mean_ms, std_ms).
    """
    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    model.eval()
    is_cuda = device == "cuda"

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
            if is_cuda:
                torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(runs):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    times_t = torch.tensor(times)
    return float(times_t.mean()), float(times_t.std())


def measure_peak_memory(model: torch.nn.Module, input_size: int, device: str):
    """Measure peak GPU memory during a forward pass (MB).

    Only meaningful on CUDA. Returns None on CPU/MPS.
    """
    if device != "cuda":
        return None

    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    model.eval()

    # Reset and measure
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    with torch.no_grad():
        _ = model(dummy)
        torch.cuda.synchronize()

    peak_bytes = torch.cuda.max_memory_allocated()
    return peak_bytes / (1024 ** 2)  # Convert to MB


def get_checkpoint_size(opt: dict) -> float | None:
    """Get checkpoint file size in MB, or None if not found."""
    ckpt_path = opt.get("path", {}).get("pretrain_model", None)
    if ckpt_path and os.path.exists(ckpt_path):
        return os.path.getsize(ckpt_path) / (1024 ** 2)
    return None


def get_psnr_from_results(exp_name: str) -> float | None:
    """Load PSNR from existing results.json."""
    results_path = os.path.join("results", exp_name, "results.json")
    if not os.path.exists(results_path):
        return None
    with open(results_path) as f:
        data = json.load(f)
    return data.get("metrics", {}).get("psnr_avg", None)


def benchmark_variant(opt: dict, input_size: int, warmup: int, runs: int,
                      device: str) -> dict:
    """Run all benchmarks for a single variant."""
    exp_name = opt.get("name", "unknown")
    display = DISPLAY_NAMES.get(exp_name, exp_name)

    print(f"\n{'─' * 60}")
    print(f"  Benchmarking: {display}  ({exp_name})")
    print(f"{'─' * 60}")

    # Build model (don't load checkpoint — architecture is enough for
    # FLOPs/latency; checkpoint is only needed for size-on-disk)
    model = build_model(opt)
    net = model.net  # The actual nn.Module (NAFNet)
    net.to(device)
    net.eval()

    # 1. Parameter count
    n_params = count_parameters(net)
    print(f"  Parameters   : {n_params:,}")

    # 2. FLOPs / MACs
    macs, thop_params = measure_flops(net, input_size, device)
    if macs is not None:
        gflops = (macs * 2) / 1e9  # FLOPs ≈ 2 × MACs
        gmacs = macs / 1e9
        print(f"  MACs         : {gmacs:.3f} G")
        print(f"  FLOPs        : {gflops:.3f} G")
    else:
        gflops = None
        gmacs = None

    # 3. Inference latency
    mean_ms, std_ms = measure_latency(net, input_size, device, warmup, runs)
    print(f"  Latency      : {mean_ms:.2f} ± {std_ms:.2f} ms  ({device})")

    # 4. Peak GPU memory
    peak_mem = measure_peak_memory(net, input_size, device)
    if peak_mem is not None:
        print(f"  Peak GPU mem : {peak_mem:.1f} MB")
    else:
        print(f"  Peak GPU mem : N/A (not CUDA)")

    # 5. Model size on disk
    ckpt_size = get_checkpoint_size(opt)
    if ckpt_size is not None:
        print(f"  Checkpoint   : {ckpt_size:.1f} MB")
    else:
        print(f"  Checkpoint   : N/A (not found)")

    # 6. PSNR and derived PSNR/GFLOP
    psnr = get_psnr_from_results(exp_name)
    psnr_per_gflop = None
    if psnr is not None and gflops is not None and gflops > 0:
        psnr_per_gflop = psnr / gflops
        print(f"  PSNR         : {psnr:.4f} dB")
        print(f"  PSNR/GFLOP   : {psnr_per_gflop:.2f} dB/GFLOP")
    elif psnr is not None:
        print(f"  PSNR         : {psnr:.4f} dB")

    return {
        "experiment": exp_name,
        "display_name": display,
        "params": n_params,
        "params_M": round(n_params / 1e6, 3),
        "macs_G": round(gmacs, 3) if gmacs is not None else None,
        "flops_G": round(gflops, 3) if gflops is not None else None,
        "latency_mean_ms": round(mean_ms, 2),
        "latency_std_ms": round(std_ms, 2),
        "latency_device": device,
        "peak_memory_MB": round(peak_mem, 1) if peak_mem is not None else None,
        "checkpoint_size_MB": round(ckpt_size, 1) if ckpt_size is not None else None,
        "psnr_dB": psnr,
        "psnr_per_gflop": round(psnr_per_gflop, 2) if psnr_per_gflop is not None else None,
        "input_size": input_size,
    }


def print_summary_table(results: list[dict]):
    """Print a formatted summary table to stdout."""
    print(f"\n{'=' * 100}")
    print(f"  EFFICIENCY BENCHMARK SUMMARY  (input: {results[0]['input_size']}×{results[0]['input_size']})")
    print(f"{'=' * 100}\n")

    # Header
    header = (
        f"{'Variant':<18} {'Params':>8} {'GMACs':>8} {'GFLOPs':>8} "
        f"{'Latency':>12} {'Peak Mem':>10} {'Ckpt':>8} "
        f"{'PSNR':>8} {'PSNR/GF':>9}"
    )
    print(header)
    print("─" * len(header))

    for r in results:
        params_str = f"{r['params_M']:.1f}M"
        macs_str = f"{r['macs_G']:.2f}" if r['macs_G'] is not None else "N/A"
        flops_str = f"{r['flops_G']:.2f}" if r['flops_G'] is not None else "N/A"
        lat_str = f"{r['latency_mean_ms']:.1f}±{r['latency_std_ms']:.1f}ms"
        mem_str = f"{r['peak_memory_MB']:.0f} MB" if r['peak_memory_MB'] is not None else "N/A"
        ckpt_str = f"{r['checkpoint_size_MB']:.0f} MB" if r['checkpoint_size_MB'] is not None else "N/A"
        psnr_str = f"{r['psnr_dB']:.2f}" if r['psnr_dB'] is not None else "N/A"
        pgf_str = f"{r['psnr_per_gflop']:.2f}" if r['psnr_per_gflop'] is not None else "N/A"

        print(
            f"{r['display_name']:<18} {params_str:>8} {macs_str:>8} {flops_str:>8} "
            f"{lat_str:>12} {mem_str:>10} {ckpt_str:>8} "
            f"{psnr_str:>8} {pgf_str:>9}"
        )

    print(f"\n  Device: {results[0]['latency_device']}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark NAFNet variants: FLOPs, latency, memory, size"
    )
    parser.add_argument(
        "--opt", type=str, default=None,
        help="Path to a single YAML config. If omitted, benchmarks all 6 canonical variants."
    )
    parser.add_argument(
        "--input_size", type=int, default=256,
        help="Spatial resolution for the dummy input (default: 256 → 256×256)"
    )
    parser.add_argument(
        "--warmup", type=int, default=100,
        help="Number of warm-up iterations before timing (default: 100)"
    )
    parser.add_argument(
        "--runs", type=int, default=500,
        help="Number of timed inference runs (default: 500)"
    )
    parser.add_argument(
        "--output", type=str, default="results/benchmark_efficiency.json",
        help="Path to save the JSON results (default: results/benchmark_efficiency.json)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"\n[Benchmark] Device: {device}")
    print(f"[Benchmark] Input : (1, 3, {args.input_size}, {args.input_size})")
    print(f"[Benchmark] Warmup: {args.warmup}, Runs: {args.runs}")

    if args.opt:
        configs = [args.opt]
    else:
        configs = [c for c in CANONICAL_CONFIGS if os.path.exists(c)]
        if not configs:
            print("[Error] No canonical config files found. Are you running from the project root?")
            sys.exit(1)
        print(f"[Benchmark] Found {len(configs)} canonical configs")

    results = []
    for cfg_path in configs:
        opt = load_config(cfg_path)
        r = benchmark_variant(opt, args.input_size, args.warmup, args.runs, device)
        results.append(r)

    # Summary table
    print_summary_table(results)

    # Save JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"benchmarks": results}, f, indent=2)
    print(f"[Benchmark] Results saved to {args.output}")


if __name__ == "__main__":
    main()
