""" 
General AP Statistics for Occupancy Heatmap
"""
from itertools import product
from typing import Dict, List, Optional

import numpy as np
from scipy import integrate
from torch import Tensor
from konductor.metadata.statistics import Statistic, STATISTICS_REGISTRY


@STATISTICS_REGISTRY.register_module("occupancy")
class Occupancy(Statistic):
    """Soft IoU and AUC for Occupancy"""

    sort_lambda = {"AUC": max, "IoU": max}

    def __init__(
        self,
        time_idxs: Optional[List[int]] = None,
        classes: Optional[List[str]] = None,
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
                f"{classname}_IoU_{timestep}",
                self.calculate_soft_iou(prediction[:, tidx], target[:, tidx]),
            )

            self._append_sample(
                f"{classname}_AUC_{timestep}",
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
                f"{classname}_IoU", self.calculate_soft_iou(prediction, target)
            )
            self._append_sample(
                f"{classname}_AUC", self.calculate_auc(prediction, target)
            )
        else:
            self.run_over_timesteps(prediction, target, timesteps, classname)

    def __call__(
        self, predictions: Dict[str, Tensor], targets: Dict[str, Tensor]
    ) -> None:
        """"""
        for idx_, name in enumerate(predictions):
            prediction = predictions[name]
            target = targets["heatmap"][:, idx_]
            self.run_over_classes(
                prediction,
                target,
                targets["time_idx"],
                f"_{name}" if name != "heatmap" else "",
            )
        self._end_idx += 1
