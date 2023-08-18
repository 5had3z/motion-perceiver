from typing import Dict, Set
from pathlib import Path
import gc

import numpy as np
import torch
from torch import Tensor
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from konductor.utilities.pbar import LivePbar

from src.model.motion_perceiver import MotionPerceiver
from .eval_common import yield_filtered_batch


@torch.inference_mode()
def inference(
    model: MotionPerceiver, batch: Dict[str, Tensor], pred_dir: Path, gt_dir: Path
) -> int:
    """Performs inference on testing data and writes prediction and ground truth
    to a file for later post processing and analysis.
    Returns number of ids written"""
    preds: Dict[str, Tensor] = model(**batch)

    if "flow" in preds:
        pred_batch = torch.cat([preds["heatmap"].unsqueeze(1), preds["flow"]], dim=1)
    else:
        pred_batch = preds["heatmap"].unsqueeze(1)

    # Write to File
    for filename, pred, gt in zip(batch["scenario_id"], pred_batch, batch["heatmap"]):
        np.save(pred_dir / f"{filename}.npy", pred.cpu().numpy())
        np.save(gt_dir / f"{filename}.npy", gt.cpu().numpy())

    return pred_batch.shape[0]


def run_export(
    model: MotionPerceiver,
    dataloader: DALIGenericIterator,
    scenario_ids: Set[str],
    pred_path: Path,
    gt_path: Path,
    batch_size: int,
):
    """Run inference on scenario ids and write
    predictions to pred_path and the ground truth to gt_path"""
    with LivePbar(total=len(scenario_ids), desc="Exporting") as pbar:
        for batch in yield_filtered_batch(dataloader, scenario_ids, batch_size):
            pbar.update(inference(model, batch[0], pred_path, gt_path))
            del batch
    gc.collect()
    torch.cuda.empty_cache()
