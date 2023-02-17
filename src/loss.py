"""Loss function and utilities for occupancy prediction for heatmap
"""
from typing import Any, Dict
from dataclasses import dataclass

import torch
from torch import nn, Tensor
from torch.nn.functional import binary_cross_entropy_with_logits

from konductor.modules.losses import REGISTRY, LossConfig, ExperimentInitConfig


class Occupancy(nn.Module):
    def __init__(self, weight: float = 1.0, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight
        self.pos_weight = torch.tensor(pos_weight).cuda()

    def forward(
        self, predictions: Dict[str, Any], targets: Dict[str, Tensor]
    ) -> Tensor:
        """"""
        loss = binary_cross_entropy_with_logits(
            predictions["heatmap"], targets["heatmap"], pos_weight=self.pos_weight
        )
        return loss


@dataclass
@REGISTRY.register_module("occupancy")
class OccupancyLoss(LossConfig):
    pos_weight: float = 1.0

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int):
        return cls(**config.criterion[idx].args)

    def get_instance(self) -> Any:
        return Occupancy(self.weight, self.pos_weight)


class OccupancyFocal(nn.Module):
    def __init__(
        self,
        weight: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2,
        pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.pos_weight = torch.tensor(pos_weight).cuda()

    def _forward_aux(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Apply loss to prediction, target pair"""
        prob = prediction.sigmoid()
        ce_loss = binary_cross_entropy_with_logits(
            prediction, target, pos_weight=self.pos_weight, reduction="none"
        )
        p_t = prob * target + (1 - prob) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        return loss.mean()

    def forward(
        self, predictions: Dict[str, Tensor], targets: Dict[str, Tensor]
    ) -> Tensor:
        """"""
        loss = torch.zeros(1).cuda()
        for idx_, name in enumerate(predictions):
            loss += self._forward_aux(predictions[name], targets["heatmap"][:, idx_])

        return loss


@dataclass
@REGISTRY.register_module("occupancy_focal")
class OccupancyFocalLoss(LossConfig):
    alpha: float = 1.0
    gamma: float = 0.25
    pos_weight: float = 1.0

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int):
        return cls(**config.criterion[idx].args)

    def get_instance(self) -> Any:
        return OccupancyFocal(self.weight, self.alpha, self.gamma, self.pos_weight)
