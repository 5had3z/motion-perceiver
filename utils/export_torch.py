from typing import Dict, List, Set, Tuple
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from konductor.data import get_dataloader, get_dataset_config
from konductor.models import get_model
from konductor.init import ExperimentInitConfig
from konductor.trainer.pbar import ProgressBar

from src.model.motion_perceiver import MotionPerceiver
from src.dataset.waymo import WaymoDatasetConfig
from evaluate import yield_filtered_batch


def scenairo_id_tensor_2_str(batch_ids: Tensor):
    scenario_str = []
    for id_char in batch_ids:
        scenario_str.append("".join([chr(c) for c in id_char]))
    return scenario_str


def scenario_id_mask(batch_ids: List[str], valid_scenarios: Set[str]) -> Tensor:
    """Batch of scenario ids are checked if they are valid, mask is returned"""
    mask = [id_str in valid_scenarios for id_str in batch_ids]
    return torch.as_tensor(mask, dtype=torch.bool)


@torch.inference_mode()
def inference(
    model: MotionPerceiver, batch: Dict[str, Tensor], pred_dir: Path, gt_dir: Path
) -> int:
    """Performs inference on testing data and writes prediction and ground truth
    to a file for later post processing and analysis.
    Returns number of ids written"""
    heatmaps: Tensor = model(**batch)["heatmap"]

    # Write to File
    for filename, pred, gt in zip(batch["scenario_id"], heatmaps, batch["heatmap"]):
        np.save(pred_dir / f"{filename}.npy", pred.cpu().numpy())
        np.save(gt_dir / f"{filename}.npy", gt[None].cpu().numpy())

    return heatmaps.shape[0]


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
    with ProgressBar(total=len(scenario_ids), desc="Exporting") as pbar:
        for batch in yield_filtered_batch(dataloader, scenario_ids, batch_size):
            pbar.update(inference(model, batch[0], pred_path, gt_path))


def initialize(
    exp_cfg: ExperimentInitConfig, args
) -> Tuple[MotionPerceiver, DALIGenericIterator]:
    """Initialise model and dataloader for prediction export"""
    exp_cfg.model[0].optimizer.args.pop("step_interval", None)

    data_cfg: WaymoDatasetConfig = get_dataset_config(exp_cfg)
    data_cfg.filter_future = True
    data_cfg.scenario_id = True
    data_cfg.use_sdc_frame = True
    data_cfg.waymo_eval_frame = True
    data_cfg.only_vehicles = True
    data_cfg.val_loader.args["batch_size"] = 1
    data_cfg.random_heatmap_count = 0
    data_cfg.heatmap_time = list(range(20, 91, 10))

    model: MotionPerceiver = get_model(exp_cfg).cuda()
    ckpt = torch.load(
        exp_cfg.work_dir / "latest.pt",
        map_location=f"cuda:{torch.cuda.current_device()}",
    )["model"]
    model.load_state_dict(ckpt)

    return model.eval(), get_dataloader(data_cfg, args.split)
