import torch
from torch.utils.data import Dataset, DataLoader


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


def build_dataloader(opt, phase="train"):
    ds_cfg = opt.get("datasets", {}).get(phase, {})
    ds_type = ds_cfg.get("type", "Synthetic")

    if ds_type in ("Synthetic", "SyntheticNoisy"):
        dataset = SyntheticNoisyDataset(
            num_samples=int(ds_cfg.get("num_samples", 400)),
            patch_size=int(ds_cfg.get("patch_size", 64)),
            noise_sigma=float(ds_cfg.get("noise_sigma", 0.1)),
        )
    else:
        raise ValueError(f"Unsupported dataset type: '{ds_type}'")

    return DataLoader(
        dataset,
        batch_size=int(ds_cfg.get("batch_size", 4)),
        shuffle=bool(ds_cfg.get("shuffle", phase == "train")),
        num_workers=int(ds_cfg.get("num_workers", 0)),
        pin_memory=bool(ds_cfg.get("pin_memory", torch.cuda.is_available())),
        drop_last=phase == "train",
    )
