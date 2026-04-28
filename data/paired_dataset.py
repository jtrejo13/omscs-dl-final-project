# ------------------------------------------------------------------------
# Adapted from NAFNet (https://github.com/megvii-research/NAFNet)
# "Simple Baselines for Image Restoration", Chen et al., ECCV 2022
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import io
import os
import random

import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _read_lmdb_keys(lmdb_path: str) -> list[str]:
    """Return sorted list of keys from an LMDB's meta_info.txt"""
    meta = os.path.join(lmdb_path, 'meta_info.txt')
    with open(meta) as f:
        return sorted(line.split('.')[0] for line in f if line.strip())


def _lmdb_decode(env: lmdb.Environment, key: str) -> np.ndarray:
    """Read one image from an open LMDB env; return float32 HWC in [0, 1]"""
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img = Image.open(io.BytesIO(buf)).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert HWC float32 numpy array to CHW float32 tensor"""
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


def paired_random_crop(img_lq: np.ndarray, img_gt: np.ndarray, patch_size: int) -> tuple:
    """Paired random crop. Ported from NAFNet/basicsr/data/transforms.py"""
    h, w = img_lq.shape[:2]
    if h < patch_size or w < patch_size:
        raise ValueError(
            f"Image ({h}x{w}) is smaller than patch_size ({patch_size})."
        )
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    return (
        img_lq[top:top + patch_size, left:left + patch_size, :],
        img_gt[top:top + patch_size, left:left + patch_size, :],
    )


def augment(imgs: list, hflip: bool = True, rotation: bool = True) -> list:
    """Random horizontal flip and 90-degree rotations. Ported from NAFNet/basicsr/data/transforms.py"""
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _aug(img):
        if hflip:
            img = img[:, ::-1, :].copy()
        if vflip:
            img = img[::-1, :, :].copy()
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_aug(img) for img in imgs]


class PairedImageDataset(Dataset):
    """Loads paired (LQ, GT) images from two LMDB databases.

    LQ and GT images are matched by key, both LMDBs must have identical
    key sets (validated at init from their meta_info.txt files).

    Args:
        lq_lmdb: Path to the LQ (noisy) LMDB directory.
        gt_lmdb: Path to the GT (clean) LMDB directory.
        patch_size: Random crop size during training. 0 = return full image.
        use_flip: Enable random horizontal flip (train phase only).
        use_rot: Enable random rotation (train phase only).
        phase: 'train', 'val', or 'test'. Augmentation only applied for 'train'.
        split: 'all' (use every key), 'val' (first split_ratio fraction after a
            seeded shuffle), or 'test' (the remainder). Same (split_ratio,
            split_seed) on both val/test sides yields disjoint, reproducible
            sets that are identical across runs and across model variants.
        split_ratio: Fraction of keys assigned to 'val'; 'test' gets 1 - ratio.
        split_seed: RNG seed for the split shuffle. Must be identical across
            val and test configs to guarantee disjointness.
    """

    def __init__(
        self,
        lq_lmdb: str,
        gt_lmdb: str,
        patch_size: int = 0,
        use_flip: bool = False,
        use_rot: bool = False,
        phase: str = 'train',
        split: str = 'all',
        split_ratio: float = 0.8,
        split_seed: int = 42,
    ):
        super().__init__()
        self.lq_lmdb = lq_lmdb
        self.gt_lmdb = gt_lmdb
        self.patch_size = patch_size
        self.use_flip = use_flip
        self.use_rot = use_rot
        self.phase = phase

        lq_keys = _read_lmdb_keys(lq_lmdb)
        gt_keys = _read_lmdb_keys(gt_lmdb)
        if lq_keys != gt_keys:
            raise ValueError(
                f"LQ and GT LMDB key sets do not match.\n"
                f"  LQ: {lq_lmdb}\n  GT: {gt_lmdb}"
            )

        if split == 'all':
            self.keys = lq_keys
        else:
            self.keys = self._apply_split(lq_keys, split, split_ratio, split_seed)
            print(
                f"[Dataset] phase={phase!r}  split={split!r}  "
                f"ratio={split_ratio}  seed={split_seed}  "
                f"size={len(self.keys)}/{len(lq_keys)}  "
                f"first_key={self.keys[0]}  last_key={self.keys[-1]}"
            )

        # Opened lazily in __getitem__
        # LMDB envs cannot be shared across forked DataLoader worker processes.
        self._lq_env: lmdb.Environment | None = None
        self._gt_env: lmdb.Environment | None = None

    @staticmethod
    def _apply_split(keys: list, split: str, ratio: float, seed: int) -> list:
        """Deterministic disjoint split. Returns sorted subset of keys"""
        if split not in ('val', 'test'):
            raise ValueError(
                f"split must be 'all', 'val', or 'test'; got {split!r}"
            )
        if not (0.0 < ratio < 1.0):
            raise ValueError(
                f"split_ratio must be in (0, 1); got {ratio}"
            )
        if len(keys) < 2:
            raise ValueError(
                f"Cannot split {len(keys)} keys into val/test."
            )

        rng = random.Random(seed)
        shuffled = list(keys)
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * ratio)
        cut = max(1, min(len(shuffled) - 1, cut))
        subset = shuffled[:cut] if split == 'val' else shuffled[cut:]
        return sorted(subset)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict:
        if self._lq_env is None:
            self._lq_env = lmdb.open(self.lq_lmdb, readonly=True, lock=False, readahead=False)
            self._gt_env = lmdb.open(self.gt_lmdb, readonly=True, lock=False, readahead=False)

        key = self.keys[idx]
        lq = _lmdb_decode(self._lq_env, key)
        gt = _lmdb_decode(self._gt_env, key)

        if self.patch_size > 0 and self.phase == 'train':
            lq, gt = paired_random_crop(lq, gt, self.patch_size)

        if self.phase == 'train' and (self.use_flip or self.use_rot):
            lq, gt = augment([lq, gt], hflip=self.use_flip, rotation=self.use_rot)

        return {'lq': _to_tensor(lq), 'gt': _to_tensor(gt), 'path': key}
