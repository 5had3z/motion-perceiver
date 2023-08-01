""" 
General AP Statistics for Occupancy Heatmap
"""
from dataclasses import dataclass
from pathlib import Path
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import l1_loss, mse_loss
from konductor.metadata.statistics import Statistic, STATISTICS_REGISTRY


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


@STATISTICS_REGISTRY.register_module("occupancy")
class Occupancy(Statistic):
    """Soft IoU and AUC for Occupancy"""

    sort_lambda = {"AUC": max, "IoU": max}

    @classmethod
    def from_config(cls, buffer_length: int, writepath: Path, **kwargs):
        time_idxs = get_time_idxs(kwargs)
        return cls(
            auc_threholds=kwargs.get("auc_thresholds", 100),
            time_idxs=time_idxs if len(time_idxs) > 0 else None,
            classes=kwargs.get("classes", None),
            time_stride=kwargs.get("time_stride", 1),
            buffer_length=buffer_length,
            writepath=writepath,
            reduce_batch=True,
        )

    def __init__(
        self,
        auc_thresholds: int = 100,
        time_idxs: List[int] | None = None,
        classes: List[str] | None = None,
        time_stride: int = 1,
        **kwargs,
    ):
        super().__init__(logger_name="OccupancyEval", **kwargs)
        self.auc_thresholds = auc_thresholds
        self.time_idxs = time_idxs
        self.time_stride = time_stride
        self.classes = classes

        # Create statistic keys
        data_keys = ["IoU", "AUC"]
        if time_idxs is not None:  # Statistic_Time
            data_keys = [
                f"{s}_{t * time_stride}" for s, t in product(data_keys, time_idxs)
            ]
        if classes is not None:  # Class_Statistic
            data_keys = [f"{c}_{s}" for c, s in product(classes, data_keys)]

        # Add Statistic Keys and dummy buffer
        for key in data_keys:
            self._statistics[key] = np.empty(self._buffer_length)

        self.reset()

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
    ) -> None:
        """Currently assume the same timesteps across batches"""
        for tidx, timestep in enumerate(timesteps[0]):
            timestep *= self.time_stride
            iou = self.calculate_soft_iou(prediction[:, tidx], target[:, tidx])
            self._append_sample(f"{classname}IoU_{timestep}", iou.mean())

            auc = self.calculate_auc(prediction[:, tidx], target[:, tidx])
            self._append_sample(f"{classname}AUC_{timestep}", auc.mean())

    def run_over_classes(
        self, prediction: Tensor, target: Tensor, timesteps: Tensor, classname: str = ""
    ) -> None:
        """Run class over timestep(s)"""
        prediction = prediction.sigmoid()

        if self.time_idxs is None:
            # squeeze channel dim on prediction if required
            prediction = prediction.squeeze(1)
            target = target.squeeze(1)

            iou = self.calculate_soft_iou(prediction, target)
            self._append_sample(f"{classname}IoU", iou.mean())

            auc = self.calculate_auc(prediction, target)
            self._append_sample(f"{classname}AUC", auc.mean())
        else:
            self.run_over_timesteps(prediction, target, timesteps, classname)

    def __call__(
        self, it: int, predictions: Dict[str, Tensor], targets: Dict[str, Tensor]
    ) -> None:
        """"""
        super().__call__(it)
        for idx_, name in enumerate(p for p in predictions if "heatmap" in p):
            prediction = predictions[name]
            target = targets["heatmap"][:, idx_]
            prefix = f"{name}_" if name != "heatmap" else ""
            self.run_over_classes(prediction, target, targets["time_idx"], prefix)


class Signal(Statistic):
    """Tracking signal prediction performance"""

    sort_lambda = {"AUC": max, "IoU": max}

    @classmethod
    def from_config(cls, buffer_length: int, writepath: Path, **kwargs):
        return cls(
            time_idxs=get_time_idxs(kwargs),
            buffer_length=buffer_length,
            writepath=writepath,
            reduce_batch=True,
        )

    def __init__(self, time_idxs: List[int], **kwargs) -> None:
        super().__init__(logger_name="SignalEval", **kwargs)
        self.time_idxs = time_idxs

        # Create statistic keys
        data_keys = [f"Acc_{t}" for t in time_idxs]

        # Add Statistic Keys and dummy buffer
        for key in data_keys:
            self._statistics[key] = np.empty(0)

        self.reset()

    @staticmethod
    def get_targets(
        targets: Tensor, mask: Tensor, timestamps: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Build the signal ground truth to target"""
        return targets[:, timestamps, :, -1], mask[:, timestamps]

    def __call__(
        self, it: int, predictions: Dict[str, Tensor], targets: Dict[str, Tensor]
    ) -> None:
        """"""
        super().__call__(it)
        target_signal, valid_signal = self.get_targets(
            targets["signals"], targets["signals_valid"], targets["time_idx"][0]
        )
        correct = predictions["signals"].argmax(dim=-1) == target_signal.long()
        correct *= valid_signal.bool()  # Mask incorrect to false
        accuracy = correct.sum(dim=[0, 2]) / valid_signal.sum(dim=[0, 2])

        for acc, tidx in zip(accuracy, targets["time_idx"][0]):
            self._append_sample(f"Acc_{tidx.item()}", acc.item())


class Flow(Statistic):
    sort_lambda = {"mse": min, "l1": min}

    @classmethod
    def from_config(cls, buffer_length: int, writepath: Path, **kwargs):
        return cls(
            time_idxs=get_time_idxs(kwargs),
            buffer_length=buffer_length,
            writepath=writepath,
            reduce_batch=True,
        )

    def __init__(self, time_idxs: List[int], **kwargs) -> None:
        super().__init__(logger_name="FlowEval", **kwargs)
        self.time_idxs = time_idxs

        # Create statistic keys
        data_keys = []
        for key in Flow.sort_lambda:
            data_keys.extend([f"{key}_{t}" for t in time_idxs])

        # Add Statistic Keys and dummy buffer
        for key in data_keys:
            self._statistics[key] = np.empty(0)

        self.reset()

    def __call__(self, it: int, prd: Dict[str, Tensor], tgt: Dict[str, Tensor]) -> None:
        """"""
        super().__call__(it)

        fn = {"mse": mse_loss, "l1": l1_loss}

        mask_time = tgt["heatmap"].transpose(0, 2)
        for key in Flow.sort_lambda:
            perf = fn[key](prd["flow"], tgt["flow"], reduction="none") * tgt["heatmap"]
            perf_time = perf.transpose(0, 2)
            for acc, mask, tidx in zip(perf_time, mask_time, tgt["time_idx"][0]):
                acc = acc.sum(dim=[0, 2, 3]) / mask.sum(dim=[0, 2, 3])
                self._append_sample(f"{key}_{tidx.item()}", acc.mean().item())
