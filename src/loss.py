"""Loss function and utilities for occupancy prediction for heatmap
"""
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import torch
from konductor.losses import REGISTRY, LossConfig
from torch import Tensor, nn
from torch.nn.functional import (
    binary_cross_entropy_with_logits,
    huber_loss,
    l1_loss,
    mse_loss,
)


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

    def get_instance(self) -> Any:
        return OccupancyBCE(**asdict(self))


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
        loss = torch.zeros(1, device=targets["heatmap"].device).squeeze()
        for idx_, name in enumerate(p for p in predictions if "heatmap" in p):
            loss += self._forward_aux(predictions[name], targets["heatmap"][:, idx_])

        return {"focal": self.weight * loss}


@dataclass
@REGISTRY.register_module("occupancy_focal")
class OccupancyFocalLoss(LossConfig):
    alpha: float = 1.0
    gamma: float = 0.25
    pos_weight: float = 1.0

    def get_instance(self) -> Any:
        return OccupancyFocal(**asdict(self))


class SignalCE(nn.CrossEntropyLoss):
    """CE between predicted and actual traffic signalling"""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(reduction="none", ignore_index=-1)
        self._weight = weight

    @staticmethod
    def get_targets(
        targets: Tensor, mask: Tensor, timestamps: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Build the signal ground truth to target"""
        return targets[:, timestamps, :, -1], mask[:, timestamps]

    def forward(
        self, preds: Dict[str, Tensor], targets: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        target_signal, valid_signal = self.get_targets(
            targets["signals"], targets["signals_valid"], targets["time_idx"][0]
        )
        loss = super().forward(
            preds["signals"].flatten(end_dim=-2), target_signal.flatten().long()
        )

        # Apply mask to zero out invalid signal indicies
        loss *= valid_signal.flatten()

        return {"signal_bce": self._weight * loss.mean()}


@dataclass
@REGISTRY.register_module("signal_prediction")
class SignalBCEConfig(LossConfig):
    def get_instance(self) -> Any:
        return SignalCE(**asdict(self))


class FlowLoss(nn.Module):
    """Apply regression loss to predicted occupancy flow"""

    def __init__(self, loss_type: str, weight=1.0, only_occupied: bool = True) -> None:
        """
        only_occupied: only apply loss where occupancy is present
        """
        super().__init__()
        self.weight = weight
        self.only_occupied = only_occupied
        self.loss_fn = {"huber": huber_loss, "mse": mse_loss, "l1": l1_loss}[loss_type]

    def forward(
        self, preds: Dict[str, Tensor], targets: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """"""
        loss: Tensor = self.loss_fn(preds["flow"], targets["flow"], reduction="none")
        if self.only_occupied:
            loss *= targets["heatmap"]

        return {"flow": self.weight * loss.mean()}


@dataclass
@REGISTRY.register_module("occupancy_flow")
class FlowLossConfig(LossConfig):
    loss_type: str = "huber"
    only_occupied: bool = True

    def get_instance(self) -> Any:
        return FlowLoss(**asdict(self))


class ConservationLoss(nn.Module):
    """Sum of gt pixels should equal sum of sigmoid logits"""

    def __init__(self, weight: float = 0) -> None:
        super().__init__()
        self.weight = weight

    def forward(
        self, preds: Dict[str, Tensor], targets: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        loss = l1_loss(
            torch.sum(preds["heatmap"].sigmoid(), dim=[-2, -1]),
            torch.sum(targets["heatmap"][:, 0], dim=[-2, -1]),
        )
        return {"conservation": self.weight * loss}


@dataclass
@REGISTRY.register_module("conservation")
class ConservationConfig(LossConfig):
    def get_instance(self) -> Any:
        return ConservationLoss(**asdict(self))
