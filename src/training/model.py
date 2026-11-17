"""Model, loss, and optimization factories for the locked Phase 4 baseline."""

from __future__ import annotations

import hashlib

import torch
from monai.losses import DiceCELoss
from monai.networks.nets import UNet

from .config import BaselineConfig


def build_model(config: BaselineConfig) -> UNet:
    return UNet(
        spatial_dims=config.spatial_dims,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        channels=config.channels,
        strides=config.strides,
        num_res_units=config.num_res_units,
        norm=config.normalization,
        act=config.activation,
        dropout=config.dropout,
    )


def build_loss(config: BaselineConfig) -> DiceCELoss:
    return DiceCELoss(
        include_background=config.loss_include_background,
        to_onehot_y=config.loss_to_onehot_y,
        softmax=config.loss_softmax,
        lambda_dice=config.lambda_dice,
        lambda_ce=config.lambda_ce,
        label_smoothing=config.label_smoothing,
    )


def build_optimizer(model: torch.nn.Module, config: BaselineConfig) -> torch.optim.Optimizer:
    if config.optimizer != "AdamW":
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: BaselineConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    if config.scheduler != "CosineAnnealingLR":
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.scheduler_t_max_epochs,
    )


def model_state_sha256(model: torch.nn.Module) -> str:
    """Hash parameters and buffers without serializing a checkpoint."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()
