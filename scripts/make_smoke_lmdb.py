"""
make_smoke_lmdb.py
------------------
Carve a small subset of the SIDD val LMDB into a local "smoke train" LMDB.

This gives the smoke test a genuine train split (different file from val),
matching the train_baseline.yml structure without requiring the full 70 GB
SIDD Medium download.

Usage
-----
    python scripts/make_smoke_lmdb.py

Output
------
    data/SIDD/train_smoke/input_crops.lmdb
    data/SIDD/train_smoke/gt_crops.lmdb
"""

import os
import lmdb

N = 80  # number of patches to carve out

PAIRS = [
    (
        "data/SIDD/val/input_crops.lmdb",
        "data/SIDD/train_smoke/input_crops.lmdb",
    ),
    (
        "data/SIDD/val/gt_crops.lmdb",
        "data/SIDD/train_smoke/gt_crops.lmdb",
    ),
]


def read_keys(lmdb_path: str) -> list[str]:
    meta = os.path.join(lmdb_path, "meta_info.txt")
    with open(meta) as f:
        return sorted(line.split(".")[0] for line in f if line.strip())


def carve(src_path: str, dst_path: str, keys: list[str]) -> None:
    os.makedirs(dst_path, exist_ok=True)
    src_env = lmdb.open(src_path, readonly=True, lock=False, readahead=False)
    dst_env = lmdb.open(dst_path, map_size=200 * 1024 * 1024)  # 200 MB

    with src_env.begin(write=False) as src_txn:
        with dst_env.begin(write=True) as dst_txn:
            for key in keys:
                value = src_txn.get(key.encode("ascii"))
                dst_txn.put(key.encode("ascii"), value)

    src_env.close()
    dst_env.close()

    # Mirror the meta_info.txt entries for the carved keys
    src_meta = os.path.join(src_path, "meta_info.txt")
    key_set = set(keys)
    lines = []
    with open(src_meta) as f:
        for line in f:
            if line.strip() and line.split(".")[0] in key_set:
                lines.append(line)

    with open(os.path.join(dst_path, "meta_info.txt"), "w") as f:
        f.writelines(lines)

    print(f"  {src_path} -> {dst_path}  ({len(keys)} patches)")


def main():
    keys = read_keys(PAIRS[0][0])[:N]
    print(f"Carving first {len(keys)} keys into smoke train LMDBs...")
    for src, dst in PAIRS:
        carve(src, dst, keys)
    print("Done.")


if __name__ == "__main__":
    main()
