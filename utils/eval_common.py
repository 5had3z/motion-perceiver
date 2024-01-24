from typing import Dict, List, Set, Tuple

import torch
from konductor.data import get_dataset_config
from konductor.init import ExperimentInitConfig
from konductor.models import get_model
from torch import Tensor

from src.dataset.common import MotionDatasetConfig
from src.dataset.interaction import InteractionConfig
from src.dataset.sdd import SDDDatasetConfig
from src.dataset.waymo import WaymoDatasetConfig
from src.model.motion_perceiver import MotionPerceiver


def scenairo_id_tensor_2_str(batch_ids: Tensor) -> List[str]:
    return [
        "".join([chr(c) for c in id_chars]).rstrip("\x00") for id_chars in batch_ids
    ]


def gather_dict(meta_batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    ret_dict = {}
    for key in meta_batch[0]:
        if key == "scenario_id":
            ret_dict[key] = sum([m[key] for m in meta_batch], [])
        else:
            ret_dict[key] = torch.cat([m[key] for m in meta_batch], dim=0)
    return ret_dict


def yield_filtered_batch(dataloader, filter_ids: Set[str], batch_thresh: int):
    """Once batch_thresh is met or exeeded, a batch is yielded, therefore the maximum
    possible batch size is the dataloader batch_size + batch_thresh - 1"""
    meta_batch = []
    for batch in dataloader:
        batch = batch[0]
        batch["scenario_id"] = scenairo_id_tensor_2_str(batch["scenario_id"])

        filt_ids: List[int] = [
            i for i, s in enumerate(batch["scenario_id"]) if s in filter_ids
        ]
        if len(filt_ids) == 0:
            continue

        for key in batch:
            if key != "scenario_id":
                batch[key] = batch[key][filt_ids]
            else:
                batch["scenario_id"] = [batch["scenario_id"][i] for i in filt_ids]

        meta_batch.append(batch)
        if len(meta_batch) >= batch_thresh:
            yield [gather_dict(meta_batch)]
            meta_batch = []

    if len(meta_batch) != 0:
        yield [gather_dict(meta_batch)]


def load_model(exp_cfg: ExperimentInitConfig) -> MotionPerceiver:
    model: MotionPerceiver = get_model(exp_cfg).cuda()
    ckpt = torch.load(
        exp_cfg.exp_path / "latest.pt",
        map_location=f"cuda:{torch.cuda.current_device()}",
    )["model"]
    model.load_state_dict(ckpt)
    return model


def initialize(
    exp_cfg: ExperimentInitConfig, no_overrides: bool = False
) -> Tuple[MotionPerceiver, MotionDatasetConfig]:
    """Initialise model and dataset for prediction export"""

    # Override occupancy roi for eval if necessary
    if exp_cfg.data[0].dataset.args.get("map_normalize", 0) == 80 and not no_overrides:
        exp_cfg.data[0].dataset.args["occupancy_roi"] = 0.5

    data_cfg: MotionDatasetConfig = get_dataset_config(exp_cfg)

    model = load_model(exp_cfg)
    model.encoder.random_input_indicies = 0

    return model, data_cfg


def apply_eval_overrides(data_cfg: MotionDatasetConfig, eval_waypoints: bool = False):
    """Apply overrides to dataset for evaluation loading"""
    data_cfg.filter_future = True
    data_cfg.random_heatmap_count = 0
    data_cfg.random_heatmap_piecewise.clear()  # Ensure piecewise random is cleared
    data_cfg.scenario_id = True

    if isinstance(data_cfg, WaymoDatasetConfig):
        data_cfg.use_sdc_frame = True
        data_cfg.waymo_eval_frame = True
        data_cfg.only_vehicles = True
        start_t = 20 // data_cfg.time_stride
        end_t = 90 // data_cfg.time_stride + 1
        stride_t = 10 // data_cfg.time_stride
        data_cfg.heatmap_time = (
            list(range(start_t, end_t, stride_t))
            if eval_waypoints
            else list(range(end_t))
        )
    elif isinstance(data_cfg, InteractionConfig):
        data_cfg.heatmap_time = list(range(40 // data_cfg.time_stride))
    elif isinstance(data_cfg, SDDDatasetConfig):
        # data_cfg.heatmap_time = [52, 76]
        data_cfg.heatmap_time = list(range(data_cfg.sequence_length))
