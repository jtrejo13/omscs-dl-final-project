# ------------------------------------------------------------------------
# Adapted from NAFNet (https://github.com/megvii-research/NAFNet)
# "Simple Baselines for Image Restoration", Chen et al., ECCV 2022
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import os
import random

import numpy as np

from PIL import Image
from torch.utils.data import Dataset


_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".PNG", ".JPG", ".JPEG"}

def scan_folder(folder: str) -> list[str]:
    """Return sorted list of image paths in folder"""
    paths = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1] in _IMG_EXTENSIONS
    ]
    return paths


def _load_image(path: str) -> np.ndarray:
    """Load image as numpy array in [0, 1]"""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def paired_random_crop(img_lq: np.ndarray, img_gt: np.ndarray, patch_size: int) -> tuple:
    """Paired random crop.

    Ported from NAFNet/basicsr/data/transforms.py.
    """
    h, w = img_lq.shape[:2]
    if h < patch_size or w < patch_size:
        raise ValueError(
            f"Image ({h}x{w}) is smaller than patch_size ({patch_size})."
            "Use a smaller patch_size or larger images."
        )
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    img_lq = img_lq[top:top + patch_size, left:left + patch_size, :]
    img_gt = img_gt[top:top + patch_size, left:left + patch_size, :]
    return img_lq, img_gt


class PairedImageDataset(Dataset):
    """Dataset that loads paired (LQ, GT) images from two folders.

    LQ and GT images are paired by sorted position — filenames may differ
    between the two directories (as in SIDD's input_crops / gt_crops layout).

    Attributes:
        lq_dir: Directory of degraded (noisy) images.
        gt_dir: Directory of clean ground-truth images.
        patch_size: Random crop size during training. 0 = return full image.
        use_flip: Enable random horizontal flip augmentation.
        use_rot: Enable random rotation augmentation.
        phase: 'train', 'val', or 'test'. Augmentation only applied for 'train'.
    """

    def __init__(
        self,
        lq_dir: str,
        gt_dir: str | None = None,
        patch_size: int = 0,
        use_flip: bool = False,
        use_rot: bool = False,
        phase: str = "train",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.use_flip = use_flip
        self.use_rot = use_rot
        self.phase = phase

        self.lq_paths = scan_folder(lq_dir)
        if not self.lq_paths:
            raise ValueError(f"No images found in lq_dir: {lq_dir}")
        
        self.gt_paths = None
        if gt_dir is not None:
            self.gt_paths = scan_folder(gt_dir)
            if len(self.gt_paths) != len(self.lq_paths):
                raise ValueError(
                    f"Number of LQ images ({len(self.lq_paths)}) does not match "
                    f"GT images ({len(self.gt_paths)}) in {gt_dir}"
                )

    def __len__(self) -> int:
        return len(self.lq_paths)

    def __getitem__(self, idx: int) -> dict:
        lq_path = self.lq_paths[idx]
        lq = _load_image(lq_path)

        gt = None
        if self.gt_paths is not None:
            gt = _load_image(self.gt_paths[idx])

        # Random crop
        # Note: This is part of training only
        if self.patch_size > 0 and self.phase == "train":
            if gt is not None:
                lq, gt = paired_random_crop(lq, gt, self.patch_size)
            else:
                h, w = lq.shape[:2]
                top = random.randint(0, h - self.patch_size)
                left = random.randint(0, w - self.patch_size)
                lq = lq[top:top + self.patch_size, left:left + self.patch_size, :]