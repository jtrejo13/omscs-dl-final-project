# ------------------------------------------------------------------------
# Adapted from NAFNet (https://github.com/megvii-research/NAFNet)
# "Simple Baselines for Image Restoration", Chen et al., ECCV 2022
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import functools

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

_reduction_modes = ["none", "mean", "sum"]


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    else:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction="mean"):
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    if weight is None or reduction == "sum":
        loss = reduce_loss(loss, reduction)
    elif reduction == "mean":
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight

    return loss


def weighted_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction="mean", **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss

    return wrapper


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction="none")


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction="none")


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss."""

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(L1Loss, self).__init__()
        if reduction not in _reduction_modes:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. "
                f"Supported ones are: {_reduction_modes}"
            )
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction
        )


class MSELoss(nn.Module):
    """MSE (L2) loss."""

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(MSELoss, self).__init__()
        if reduction not in _reduction_modes:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. "
                f"Supported ones are: {_reduction_modes}"
            )
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction
        )


class PSNRLoss(nn.Module):
    """PSNR-based loss. Returns a negative value; minimizing it maximizes PSNR.

    The logged loss during training will be a small negative number (e.g. -38.0),
    which corresponds to a PSNR of ~38 dB. This is expected behavior.
    """

    def __init__(self, loss_weight=1.0, reduction="mean", toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == "mean"
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0

            pred, target = pred / 255.0, target / 255.0
        assert len(pred.size()) == 4

        return (
            self.loss_weight
            * self.scale
            * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        )
