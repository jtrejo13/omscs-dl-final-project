import torch
from torch.utils.data import Dataset, DataLoader
from .paired_dataset import PairedImageDataset


class SyntheticNoisyDataset(Dataset):
    def __init__(self, num_samples=400, patch_size=64, noise_sigma=0.1, in_channels=3):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.noise_sigma = noise_sigma
        self.in_channels = in_channels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        gt = torch.rand(self.in_channels, self.patch_size, self.patch_size)
        lq = (gt + torch.randn_like(gt) * self.noise_sigma).clamp(0.0, 1.0)
        return {"lq": lq, "gt": gt, "path": f"synthetic_{idx}"}


def build_dataloader(opt, phase="train", worker_init_fn=None, generator=None):
    ds_cfg = opt.get("datasets", {}).get(phase, {})
    ds_type = ds_cfg.get("type", "Synthetic")

    if ds_type in ("Synthetic", "SyntheticNoisyDataset"):
        dataset = SyntheticNoisyDataset(
            num_samples=int(ds_cfg.get("num_samples", 400)),
            patch_size=int(ds_cfg.get("patch_size", 64)),
            noise_sigma=float(ds_cfg.get("noise_sigma", 0.1)),
        )
    elif ds_type == "PairedImage":
        dataset = PairedImageDataset(
            lq_lmdb=ds_cfg["lq_lmdb"],
            gt_lmdb=ds_cfg["gt_lmdb"],
            patch_size=int(ds_cfg.get("patch_size", 0)),
            use_flip=bool(ds_cfg.get("use_flip", False)),
            use_rot=bool(ds_cfg.get("use_rot", False)),
            phase=phase,
            split=ds_cfg.get("split", "all"),
            split_ratio=float(ds_cfg.get("split_ratio", 0.8)),
            split_seed=int(ds_cfg.get("split_seed", 42)),
        )
        num_samples = ds_cfg.get("num_samples")
        if num_samples is not None:
            n = min(int(num_samples), len(dataset))
            dataset = torch.utils.data.Subset(dataset, range(n))
    else:
        raise ValueError(f"Unsupported dataset type: '{ds_type}'")

    return DataLoader(
        dataset,
        batch_size=int(ds_cfg.get("batch_size", 4)),
        shuffle=bool(ds_cfg.get("shuffle", phase == "train")),
        num_workers=int(ds_cfg.get("num_workers", 0)),
        pin_memory=bool(ds_cfg.get("pin_memory", torch.cuda.is_available())),
        drop_last=phase == "train",
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
