"""Loss function and utilities for occupancy prediction for heatmap
"""
from typing import Any, Dict
from dataclasses import dataclass, asdict

import torch
from torch import nn, Tensor
from torch.nn.functional import binary_cross_entropy_with_logits

from konductor.modules.losses import REGISTRY, LossConfig, ExperimentInitConfig
from konductor.modules.init import ModuleInitConfig


class OccupancyBCE(nn.Module):
    def __init__(self, weight: float = 1.0, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight
        self.pos_weight = torch.tensor(pos_weight).cuda()

    def forward(
        self, predictions: Dict[str, Any], targets: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """"""
        loss = binary_cross_entropy_with_logits(
            predictions["heatmap"], targets["heatmap"], pos_weight=self.pos_weight
        )
        return {"bce": self.weight * loss}


@dataclass
@REGISTRY.register_module("occupancy_bce")
class OccupancyLoss(LossConfig):
    pos_weight: float = 1.0

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int):
        return super().from_config(config, idx, names=["bce"])

    def get_instance(self) -> Any:
        kwargs = asdict(self)
        del kwargs["names"]
        return OccupancyBCE(**kwargs)


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
        loss = ce_loss * ((1 - p_t + torch.finfo(prob.dtype).eps) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        return loss.mean()

    def forward(
        self, predictions: Dict[str, Tensor], targets: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """"""
        loss = torch.zeros(1).cuda()
        for idx_, name in enumerate(predictions):
            loss += self._forward_aux(predictions[name], targets["heatmap"][:, idx_])

        return {"focal": self.weight * loss}


@dataclass
@REGISTRY.register_module("occupancy_focal")
class OccupancyFocalLoss(LossConfig):
    alpha: float = 1.0
    gamma: float = 0.25
    pos_weight: float = 1.0

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int):
        return super().from_config(config, idx, names=["focal"])

    def get_instance(self) -> Any:
        kwargs = asdict(self)
        del kwargs["names"]
        return OccupancyFocal(**kwargs)
