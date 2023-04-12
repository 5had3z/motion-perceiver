""" 
General AP Statistics for Occupancy Heatmap
"""
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
from scipy import integrate
from torch import Tensor
from konductor.metadata.statistics import Statistic, STATISTICS_REGISTRY


@STATISTICS_REGISTRY.register_module("occupancy")
class Occupancy(Statistic):
    """Soft IoU and AUC for Occupancy"""

    sort_lambda = {"AUC": max, "IoU": max}

    @classmethod
    def from_config(cls, buffer_length: int, writepath: Path, **kwargs):
        time_idxs = set()
        if "random_heatmap_minmax" in kwargs:
            _min, _max = kwargs["random_heatmap_minmax"]
            time_idxs.update(set(range(_min, _max + 1)))
        if "heatmap_time" in kwargs:
            time_idxs.update(set(kwargs["heatmap_time"]))

        return cls(
            time_idxs=sorted(list(time_idxs)) if len(time_idxs) > 0 else None,
            classes=kwargs.get("classes", None),
            buffer_length=buffer_length,
            writepath=writepath,
            reduce_batch=True,
        )

    def __init__(
        self,
        time_idxs: List[int] | None = None,
        classes: List[str] | None = None,
        **kwargs,
    ):
        super().__init__(logger_name="OccupancyEval", **kwargs)
        self.time_idxs = time_idxs
        self.classes = classes

        # Create statistic keys
        data_keys = ["IoU", "AUC"]
        if time_idxs is not None:  # Statistic_Time
            data_keys = [f"{s}_{t}" for s, t in product(data_keys, time_idxs)]
        if classes is not None:  # Class_Statistic
            data_keys = [f"{c}_{s}" for c, s in product(classes, data_keys)]

        # Add Statistic Keys and dummy buffer
        for key in data_keys:
            self._statistics[key] = np.empty(0)

        self.reset()

    def calculate_soft_iou(self, pred: Tensor, target: Tensor) -> np.ndarray:
        """Calculates heatmap iou"""
        soft_intersection = (pred * target).sum(dim=(1, 2))
        soft_union = (pred + target - pred * target).sum(dim=(1, 2))
        soft_iou = (soft_intersection / soft_union).cpu().numpy()
        return soft_iou

    def calculate_auc(self, pred: Tensor, target: Tensor) -> np.ndarray:
        """Calculate heatmap auc"""
        target_binary = target.bool()

        # ensure 0,0 -> 1,1 with 1 and 0 thresholds
        thresholds = np.concatenate(
            [
                np.linspace(1, 0.8, 21),
                np.linspace(0.7, 0.3, 5),
                np.linspace(0.20, 0, 21),
            ]
        )
        # thresholds = np.linspace(1, 0, 100)
        tpr = np.empty((len(thresholds), target.shape[0]))
        fpr = np.empty((len(thresholds), target.shape[0]))
        for idx, threshold in enumerate(thresholds):
            pred_binary = pred > threshold

            fn = (~pred_binary & target_binary).sum(dim=(1, 2))
            tp = (pred_binary & target_binary).sum(dim=(1, 2))
            tpr[idx] = (tp / (tp + fn)).cpu().numpy()

            fp = (pred_binary & ~target_binary).sum(dim=(1, 2))
            tn = (~pred_binary & ~target_binary).sum(dim=(1, 2))
            fpr[idx] = (fp / (tn + fp)).cpu().numpy()

        # transpose to batch first
        tpr = tpr.transpose(1, 0)
        fpr = fpr.transpose(1, 0)

        # handle nans by setting to 1, if there is no tp in gt then tpr has to be 1
        tpr[np.isnan(tpr)] = 1

        batch_auc = np.empty((target.shape[0]))
        for b_idx, b_tpr, b_fpr in zip(range(target.shape[0]), tpr, fpr):
            batch_auc[b_idx] = integrate.cumulative_trapezoid(b_tpr, b_fpr)[-1]

        return batch_auc

    def run_over_timesteps(
        self, prediction: Tensor, target: Tensor, timesteps: Tensor, classname: str = ""
    ) -> None:
        """Currently assume the same timesteps across batches"""
        for tidx, timestep in enumerate(timesteps[0]):
            self._append_sample(
                f"{classname}IoU_{timestep}",
                self.calculate_soft_iou(prediction[:, tidx], target[:, tidx]),
            )

            self._append_sample(
                f"{classname}AUC_{timestep}",
                self.calculate_auc(prediction[:, tidx], target[:, tidx]),
            )

    def run_over_classes(
        self, prediction: Tensor, target: Tensor, timesteps: Tensor, classname: str = ""
    ) -> None:
        """Run class over timestep(s)"""
        prediction = prediction.sigmoid()

        if self.time_idxs is None:
            # squeeze channel dim on prediction if required
            prediction = prediction.squeeze(1)
            target = target.squeeze(1)

            self._append_sample(
                f"{classname}IoU", self.calculate_soft_iou(prediction, target)
            )
            self._append_sample(
                f"{classname}AUC", self.calculate_auc(prediction, target)
            )
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
            self.run_over_classes(
                prediction,
                target,
                targets["time_idx"],
                f"{name}_" if name != "heatmap" else "",
            )


class Signal(Statistic):
    """Tracking signal prediction performance"""

    sort_lambda = {"AUC": max, "IoU": max}

    @classmethod
    def from_config(cls, buffer_length: int, writepath: Path, **kwargs):
        time_idxs = set()
        if "random_heatmap_minmax" in kwargs:
            _min, _max = kwargs["random_heatmap_minmax"]
            time_idxs.update(set(range(_min, _max + 1)))
        if "heatmap_time" in kwargs:
            time_idxs.update(set(kwargs["heatmap_time"]))

        return cls(
            time_idxs=sorted(list(time_idxs)),
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
