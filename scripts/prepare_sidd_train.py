"""
prepare_sidd_train.py
---------------------
Download SIDD Medium, crop to 512×512 patches, and build the train LMDBs.

Adapted from NAFNet/scripts/data_preparation/sidd.py (megvii-research/NAFNet).
Removes the BasicSR and cv2 dependencies — uses PIL and lmdb directly.

Steps
-----
1. Download SIDD Medium Srgb (~70 GB) from Google Drive to data/SIDD/raw/
2. Crop _NOISY and _GT images to 512×512 patches (step=384, matching NAFNet)
3. Build data/SIDD/train/input_crops.lmdb and gt_crops.lmdb
4. Delete intermediate PNG crops (raw images stay so the script is re-runnable)

Usage
-----
    python scripts/prepare_sidd_train.py

Output
------
    data/SIDD/train/input_crops.lmdb
    data/SIDD/train/gt_crops.lmdb
"""

import io
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

GDRIVE_ID = "1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw"
RAW_DIR   = Path("data/SIDD/raw")

CROP_SIZE = 512
STEP      = 384
N_THREAD  = 8
COMPRESS  = 1  # PNG compression level (1 = fast; matches NAFNet default)

PAIRS = [
    # (keyword, crops_dir, lmdb_path)
    ("_NOISY", Path("data/SIDD/train/input_crops"), Path("data/SIDD/train/input_crops.lmdb")),
    ("_GT",    Path("data/SIDD/train/gt_crops"),    Path("data/SIDD/train/gt_crops.lmdb")),
]

# ── Step 1: Download ──────────────────────────────────────────────────────────

def download():
    if any(RAW_DIR.rglob("*.PNG")):
        print(f"Raw images found in {RAW_DIR} — skipping download.")
        return

    import gdown
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR / "SIDD_Medium_Srgb.zip"

    if not archive.exists():
        print("Downloading SIDD Medium (~70 GB) …")
        url = f"https://drive.google.com/file/d/{GDRIVE_ID}/view"
        gdown.download(url=url, output=str(archive), quiet=False, fuzzy=True)

    print("Extracting archive …")
    shutil.unpack_archive(str(archive), str(RAW_DIR))
    print("Extraction done.")

# ── Step 2: Crop ─────────────────────────────────────────────────────────────

def scan_sidd(root: Path, keyword: str) -> list[Path]:
    """Return sorted list of PNG files whose name contains keyword."""
    return sorted(p for p in root.rglob("*.PNG") if keyword in p.name)


def _crop_worker(args):
    """Crop a single image into 512×512 patches; save as PNG files."""
    path, save_dir, keyword = args
    img = np.array(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]

    # Stem after removing the keyword (e.g. _NOISY or _GT) so paired images
    # get the same base name, matching NAFNet's key-alignment convention.
    stem = path.stem.replace(keyword, "")

    h_space = list(range(0, h - CROP_SIZE + 1, STEP))
    if h - (h_space[-1] + CROP_SIZE) > 0:
        h_space.append(h - CROP_SIZE)

    w_space = list(range(0, w - CROP_SIZE + 1, STEP))
    if w - (w_space[-1] + CROP_SIZE) > 0:
        w_space.append(w - CROP_SIZE)

    idx = 0
    for top in h_space:
        for left in w_space:
            idx += 1
            patch = img[top:top + CROP_SIZE, left:left + CROP_SIZE]
            out = save_dir / f"{stem}_s{idx:03d}.png"
            Image.fromarray(patch).save(out, format="PNG", compress_level=COMPRESS)


def crop_images(keyword: str, crops_dir: Path):
    if any(crops_dir.glob("*.png")):
        print(f"Crops exist at {crops_dir} — skipping.")
        return

    imgs = scan_sidd(RAW_DIR, keyword)
    if not imgs:
        raise FileNotFoundError(
            f"No images with keyword '{keyword}' found under {RAW_DIR}. "
            "Make sure the download and extraction completed successfully."
        )

    crops_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cropping {len(imgs)} {keyword.strip('_')} images → {crops_dir} …")
    args = [(p, crops_dir, keyword) for p in imgs]
    with Pool(N_THREAD) as pool:
        list(tqdm(pool.imap_unordered(_crop_worker, args), total=len(imgs)))

# ── Step 3: Build LMDB ───────────────────────────────────────────────────────

def build_lmdb(crops_dir: Path, lmdb_path: Path):
    paths = sorted(crops_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG crops found in {crops_dir}")

    # Estimate map size from the first image (×10 headroom)
    sample = paths[0].read_bytes()
    map_size = len(sample) * len(paths) * 10

    print(f"Building {lmdb_path}  ({len(paths):,} patches) …")
    env = lmdb.open(str(lmdb_path), map_size=map_size)
    meta_lines = []
    txn = env.begin(write=True)

    for i, path in enumerate(tqdm(paths)):
        key = path.stem
        img = Image.open(path)
        w, h = img.size
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=COMPRESS)
        txn.put(key.encode("ascii"), buf.getvalue())
        meta_lines.append(f"{key}.png ({h},{w},3) {COMPRESS}\n")

        if i % 5000 == 0 and i > 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()

    with open(lmdb_path / "meta_info.txt", "w") as f:
        f.writelines(meta_lines)

    print(f"  → {lmdb_path}  ({len(paths):,} entries)")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    download()

    for keyword, crops_dir, lmdb_path in PAIRS:
        crop_images(keyword, crops_dir)

    for keyword, crops_dir, lmdb_path in PAIRS:
        if lmdb_path.exists():
            print(f"{lmdb_path} exists — skipping.")
            continue
        build_lmdb(crops_dir, lmdb_path)

    # Remove intermediate PNG crops — LMDBs are the only artifact we need.
    for _, crops_dir, _ in PAIRS:
        if crops_dir.exists():
            shutil.rmtree(crops_dir)
            print(f"Removed intermediate crops: {crops_dir}")

    print("\nDone. Train LMDBs:")
    for _, _, lmdb_path in PAIRS:
        size_gb = sum(f.stat().st_size for f in lmdb_path.iterdir()) / 1e9
        print(f"  {lmdb_path}  ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
