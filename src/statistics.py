""" 
General AP Statistics for Occupancy Heatmap
"""
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Tuple, Callable

import numpy as np
import torch
from konductor.data import get_dataset_properties
from konductor.init import ExperimentInitConfig
from konductor.metadata import Statistic
from torch import Tensor


@dataclass
class Confusion:
    """Batch-Last Confusion Array Shape: (thresh, batchidx)"""

    @classmethod
    def preallocate(cls, batch: int, thresholds: int, device=None):
        data = torch.empty((thresholds, batch), dtype=torch.float32, device=device)
        return cls(data, data.clone(), data.clone(), data.clone())

    tp: Tensor
    fp: Tensor
    tn: Tensor
    fn: Tensor

    @property
    def device(self):
        """Device tensors are currently on"""
        return self.tp.device


def _div_no_nan(a: Tensor, b: Tensor) -> Tensor:
    """Divide and set nan/inf values to zero"""
    c = a / b
    c[~torch.isfinite(c)] = 0
    return c


def get_time_idxs(kwargs: Dict[str, Any]) -> List[int]:
    """Get time idxs from dataset properties"""
    time_idxs = set()
    if kwargs.get("random_heatmap_minmax", None) is not None:
        _min, _max = kwargs["random_heatmap_minmax"]
        time_idxs.update(set(range(_min, _max + 1)))

    if len(kwargs.get("random_heatmap_piecewise", [])) > 0:
        for pargs in kwargs["random_heatmap_piecewise"]:
            time_idxs.update(
                set(range(pargs["min"], pargs["max"] + 1, pargs["stride"]))
            )

    if "heatmap_time" in kwargs:
        time_idxs.update(set(kwargs["heatmap_time"]))

    return sorted(time_idxs)


class Occupancy(Statistic):
    """Soft IoU and AUC for Occupancy"""

    @classmethod
    def from_config(cls, cfg: ExperimentInitConfig, **extras):
        props = get_dataset_properties(cfg)
        time_idxs = get_time_idxs(props)
        return cls(
            auc_thresholds=extras.get("auc_thresholds", 100),
            time_idxs=time_idxs if len(time_idxs) > 0 else None,
            classes=props.get("classes", None),
            time_stride=props.get("time_stride", 1),
        )

    def get_keys(self) -> List[str]:
        """Keys are Class_Stat_Time if Class and Time are not None"""
        keys = ["IoU", "AUC"]
        if self.time_idxs is not None:
            keys = [
                f"{s}_{t * self.time_stride}" for s, t in product(keys, self.time_idxs)
            ]
        if self.classes is not None:
            keys = [f"{c}_{s}" for c, s in product(self.classes, keys)]
        return keys

    def __init__(
        self,
        auc_thresholds: int = 100,
        time_idxs: List[int] | None = None,
        classes: List[str] | None = None,
        time_stride: int = 1,
    ):
        self.auc_thresholds = auc_thresholds
        self.time_idxs = time_idxs
        self.time_stride = time_stride
        self.classes = classes

    def calculate_soft_iou(self, pred: Tensor, target: Tensor) -> np.ndarray:
        """Calculates heatmap iou"""
        soft_intersection = (pred * target).sum(dim=(1, 2))
        soft_union = (pred + target - pred * target).sum(dim=(1, 2))
        soft_iou = (soft_intersection / soft_union).cpu().numpy()
        return soft_iou

    def make_thresholds(self) -> np.ndarray:
        # ensure 0,0 -> 1,1 with 1 and 0 thresholds
        # thresholds = np.concatenate(
        #     [
        #         np.linspace(1, 0.8, 21),
        #         np.linspace(0.7, 0.3, 5),
        #         np.linspace(0.20, 0, 21),
        #     ]
        # )

        thresh = np.linspace(0, 1, self.auc_thresholds, dtype=np.float32)

        # Go beyond 0,1 to capture float rounding issues
        thresh[0] = -np.finfo(thresh.dtype).eps
        thresh[-1] = 1 + np.finfo(thresh.dtype).eps
        return thresh

    def calculate_confusion(self, pred: Tensor, target: Tensor) -> Confusion:
        """"""
        target_binary = target.bool()
        thresholds = self.make_thresholds()
        conf = Confusion.preallocate(pred.shape[0], thresholds.shape[0], pred.device)

        # Thresholds should ordered 0 -> 1
        for idx, threshold in enumerate(thresholds):
            pred_binary: Tensor = pred > threshold
            conf.fn[idx] = (~pred_binary & target_binary).sum(dim=(1, 2))
            conf.tp[idx] = (pred_binary & target_binary).sum(dim=(1, 2))
            conf.fp[idx] = (pred_binary & ~target_binary).sum(dim=(1, 2))
            conf.tn[idx] = (~pred_binary & ~target_binary).sum(dim=(1, 2))

        return conf

    def interpolate_pr_auc(self, confusion: Confusion) -> np.ndarray:
        """From Keras PR AUC Interpolation"""
        zero_ = torch.tensor(0, device=confusion.device)

        dtp = confusion.tp[:-1] - confusion.tp[1:]
        p = confusion.tp + confusion.fp
        dp = p[:-1] - p[1:]
        prec_slope = _div_no_nan(dtp, torch.maximum(dp, zero_))
        intercept = confusion.tp[1:] - prec_slope * p[1:]

        safe_p_ratio = torch.where(
            torch.logical_and(p[:-1] > 0, p[1:] > 0),
            _div_no_nan(p[:-1], torch.maximum(p[1:], zero_)),
            torch.ones_like(p[1:]),
        )

        pr_auc_increment = _div_no_nan(
            prec_slope * (dtp + intercept * torch.log(safe_p_ratio)),
            torch.maximum(confusion.tp[1:] + confusion.fn[1:], zero_),
        )

        return pr_auc_increment.sum(dim=0).cpu().numpy()

    def calculate_auc(self, pred: Tensor, target: Tensor) -> np.ndarray:
        """Calculate heatmap auc"""
        conf = self.calculate_confusion(pred, target)
        auc = self.interpolate_pr_auc(conf)
        return auc

    def run_over_timesteps(
        self, prediction: Tensor, target: Tensor, timesteps: Tensor, classname: str = ""
    ) -> Dict[str, float]:
        """Currently assume the same timesteps across batches"""
        result: Dict[str, float] = {}
        for tidx, timestep in enumerate(timesteps[0]):
            timestep *= self.time_stride
            iou = self.calculate_soft_iou(prediction[:, tidx], target[:, tidx])
            result[f"{classname}IoU_{timestep}"] = iou.mean()

            auc = self.calculate_auc(prediction[:, tidx], target[:, tidx])
            result[f"{classname}AUC_{timestep}"] = auc.mean()

        return result

    def run_over_classes(
        self, prediction: Tensor, target: Tensor, timesteps: Tensor, classname: str = ""
    ) -> Dict[str, float]:
        """Run class over timestep(s)"""
        prediction = prediction.sigmoid()

        if self.time_idxs is None:
            result: Dict[str, float] = {}

            # squeeze channel dim on prediction if required
            prediction = prediction.squeeze(1)
            target = target.squeeze(1)

            iou = self.calculate_soft_iou(prediction, target)
            result[f"{classname}IoU"] = iou.mean()

            auc = self.calculate_auc(prediction, target)
            result[f"{classname}AUC"] = auc.mean()
        else:
            result = self.run_over_timesteps(prediction, target, timesteps, classname)

        return result

    def __call__(
        self, predictions: Dict[str, Tensor], targets: Dict[str, Tensor]
    ) -> Dict[str, float]:
        """"""
        result = {}
        for idx_, name in enumerate(p for p in predictions if "heatmap" in p):
            prediction = predictions[name]
            target = targets["heatmap"][:, idx_]
            prefix = f"{name}_" if name != "heatmap" else ""
            result.update(
                self.run_over_classes(prediction, target, targets["time_idx"], prefix)
            )
        return result


class Signal(Statistic):
    """Tracking signal prediction performance"""

    @classmethod
    def from_config(cls, cfg: ExperimentInitConfig, **extras):
        props = get_dataset_properties(cfg)
        return cls(time_idxs=get_time_idxs(props))

    def get_keys(self) -> List[str]:
        return [f"Acc_{t}" for t in self.time_idxs]

    def __init__(self, time_idxs: List[int]) -> None:
        self.time_idxs = time_idxs

    @staticmethod
    def get_targets(
        targets: Tensor, mask: Tensor, timestamps: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Build the signal ground truth to target"""
        return targets[:, timestamps, :, -1], mask[:, timestamps]

    def __call__(
        self, predictions: Dict[str, Tensor], targets: Dict[str, Tensor]
    ) -> None:
        """"""
        target_signal, valid_signal = self.get_targets(
            targets["signals"], targets["signals_valid"], targets["time_idx"][0]
        )
        correct = predictions["signals"].argmax(dim=-1) == target_signal.long()
        correct *= valid_signal.bool()  # Mask incorrect to false
        accuracy = correct.sum(dim=[0, 2]) / valid_signal.sum(dim=[0, 2])

        ret: Dict[str, float] = {}
        for acc, tidx in zip(accuracy, targets["time_idx"][0]):
            ret[f"Acc_{tidx.item()}"] = acc.item()


class Flow(Statistic):
    @staticmethod
    def l1(pred: Tensor, tgt: Tensor, mask: Tensor) -> Tensor:
        return torch.abs(pred - tgt) * mask

    @staticmethod
    def mse(pred, tgt, mask) -> Tensor:
        return torch.norm(pred - tgt, dim=1, keepdim=True) * mask

    _fn: dict[str, Callable[[Tensor, Tensor, Tensor], Tensor]] = {
        "mse": mse,
        "l1": l1,
    }

    @classmethod
    def from_config(cls, cfg: ExperimentInitConfig, **extras):
        props = get_dataset_properties(cfg)
        return cls(time_idxs=get_time_idxs(props))

    def get_keys(self) -> List[str]:
        return [f"{k}_{t}" for k, t in product(self._fn.keys(), self.time_idxs)]

    def __init__(self, time_idxs: List[int]) -> None:
        self.time_idxs = time_idxs

    def __call__(self, prd: Dict[str, Tensor], tgt: Dict[str, Tensor]):
        """"""
        res: Dict[str, float] = {}

        mask_time = tgt["heatmap"].transpose(0, 2)
        for key, func in self._fn.items():
            perf = func(prd["flow"], tgt["flow"], tgt["heatmap"])
            perf_time = perf.transpose(0, 2)
            for acc, mask, tidx in zip(perf_time, mask_time, tgt["time_idx"][0]):
                acc = acc.sum(dim=[0, 2, 3]) / mask.sum(dim=[0, 2, 3])
                res[f"{key}_{tidx.item()}"] = acc.mean().item()

        return res
