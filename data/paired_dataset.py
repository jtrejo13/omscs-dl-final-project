# ------------------------------------------------------------------------
# Adapted from NAFNet (https://github.com/megvii-research/NAFNet)
# "Simple Baselines for Image Restoration", Chen et al., ECCV 2022
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import os
from torch.utils.data import Dataset


_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg"}

def scan_folder(folder: str) -> list[str]:
    """Return list of image paths in folder"""
    paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1] in _IMG_EXTENSIONS
    ]
    return paths


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

    def __len__(self) -> int:
        return len(self.lq_paths)

    def __getitem__(self, idx: int) -> dict:
        return idx