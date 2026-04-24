"""
DummyRestorationModel
"""

import torch
import torch.nn as nn
import torch.optim as optim


class DummyCNN(nn.Module):
    """3-layer residual conv network, depth=16, kernel=3"""

    def __init__(self, in_channels: int = 3, mid_channels: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class DummyRestorationModel:
    """
    Wrapper that mirrors the interface used by train.py / test.py.
    """

    def __init__(self, opt: dict):
        self.opt = opt
        from utils import get_device

        self.device = torch.device(get_device())

        # network
        net_cfg = opt.get("model", {})
        in_ch = net_cfg.get("in_channels", 3)
        mid_ch = net_cfg.get("mid_channels", 16)
        self.net = DummyCNN(in_ch, mid_ch).to(self.device)

        # loss
        self.criterion = nn.L1Loss()

        # optimizer
        train_cfg = opt.get("train", {})
        lr = float(train_cfg.get("lr", 2e-4))
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.lq = None  # noisy input
        self.gt = None  # ground-truth
        self.pred = None  # prediction

    def feed_data(self, data: dict):
        """Move a batch from the dataloader onto the model's device"""
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)

    def optimize(self) -> dict:
        """Train step"""
        self.net.train()
        self.optimizer.zero_grad()
        self.pred = self.net(self.lq)
        loss = self.criterion(self.pred, self.gt)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def test(self):
        """Forward pass"""
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
            },
            path,
        )
        print(f"[DummyModel] Checkpoint saved: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"[DummyModel] Checkpoint loaded: {path}")
