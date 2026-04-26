# ------------------------------------------------------------------------
# Adapted from NAFNet (https://github.com/megvii-research/NAFNet)
# "Simple Baselines for Image Restoration", Chen et al., ECCV 2022
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
import torch.optim as optim

from models.archs.nafnet_arch import NAFNet
from models.losses import PSNRLoss
from utils import get_device


class BaselineNAFNet:
    """NAFNet baseline for image denoising on SIDD.

    Architecture: NAFNet (U-Net with NAFBlocks: SimpleGate + SCA + LayerNorm2d).
    Loss: PSNRLoss (negative PSNR; minimizing it = maximizing PSNR).
    Optimizer: AdamW with iteration-based CosineAnnealingLR.

    Logged loss values will be small negatives (e.g. -38.0 ≈ 38 dB PSNR).
    """

    def __init__(self, opt: dict, block_cls=None):
        self.opt = opt
        self.device = torch.device(get_device())

        net_cfg = opt.get("model", {})
        self.net = NAFNet(
            img_channel=net_cfg.get("img_channel", 3),
            width=net_cfg.get("width", 64),
            middle_blk_num=net_cfg.get("middle_blk_num", 12),
            enc_blk_nums=net_cfg.get("enc_blk_nums", [2, 2, 4, 8]),
            dec_blk_nums=net_cfg.get("dec_blk_nums", [2, 2, 2, 2]),
            block_cls=block_cls,
        ).to(self.device)

        train_cfg = opt.get("train", {})
        lr = float(train_cfg.get("lr", 1e-3))
        weight_decay = float(train_cfg.get("weight_decay", 0.0))
        betas = tuple(train_cfg.get("betas", [0.9, 0.9]))
        self.optimizer = optim.AdamW(
            self.net.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
        )

        # Iteration-based cosine annealing; scheduler.step() is called inside
        # optimize() so it advances once per iteration, not per epoch.
        scheduler_cfg = train_cfg.get("scheduler", {})
        t_max = int(scheduler_cfg.get("T_max", 400000))
        eta_min = float(scheduler_cfg.get("eta_min", 1e-7))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=t_max, eta_min=eta_min
        )

        loss_cfg = train_cfg.get("loss", {})
        loss_type = loss_cfg.get("type", "PSNRLoss")
        if loss_type == "PSNRLoss":
            self.criterion = PSNRLoss(
                loss_weight=float(loss_cfg.get("loss_weight", 1.0))
            ).to(self.device)
        else:
            self.criterion = torch.nn.L1Loss()

        self.use_grad_clip = bool(train_cfg.get("use_grad_clip", True))

        self.lq = None
        self.gt = None
        self.pred = None

    def feed_data(self, data: dict):
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)

    def optimize(self) -> dict:
        self.net.train()
        self.optimizer.zero_grad()
        self.pred = self.net(self.lq)
        loss = self.criterion(self.pred, self.gt)
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.01)
        self.optimizer.step()
        self.scheduler.step()
        return {"loss": loss.item()}

    def test(self):
        self.net.eval()
        with torch.no_grad():
            self.pred = self.net(self.lq)

    def get_current_visuals(self) -> dict:
        out = {"lq": self.lq.detach().cpu()}
        if self.pred is not None:
            out["pred"] = self.pred.detach().cpu()
        if self.gt is not None:
            out["gt"] = self.gt.detach().cpu()
        return out

    def save(self, path: str):
        torch.save(
            {
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            path,
        )
        print(f"[BaselineNAFNet] Checkpoint saved: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        # Official NAFNet checkpoints store weights under "params"
        if "params" in ckpt:
            self.net.load_state_dict(ckpt["params"], strict=True)
        elif "net" in ckpt:
            self.net.load_state_dict(ckpt["net"])
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler"])
        else:
            self.net.load_state_dict(ckpt, strict=False)
        print(f"[BaselineNAFNet] Checkpoint loaded: {path}")
