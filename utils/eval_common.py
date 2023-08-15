from typing import Dict, List, Tuple, Set


import torch
from torch import Tensor
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from konductor.data import get_dataloader, get_dataset_config
from konductor.models import get_model
from konductor.init import ExperimentInitConfig

from src.dataset.common import MotionDatasetConfig
from src.dataset.waymo import WaymoDatasetConfig
from src.dataset.interaction import InteractionConfig
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
        exp_cfg.work_dir / "latest.pt",
        map_location=f"cuda:{torch.cuda.current_device()}",
    )["model"]
    model.load_state_dict(ckpt)
    return model


def initialize(
    exp_cfg: ExperimentInitConfig,
    split: str,
    no_overrides: bool = False,
    eval_waypoints: bool = False,
) -> Tuple[MotionPerceiver, DALIGenericIterator]:
    """Initialise model and dataloader for prediction export"""

    # Override occupancy roi for eval if necessary
    if exp_cfg.data[0].dataset.args.get("map_normalize", 0) == 80 and not no_overrides:
        exp_cfg.data[0].dataset.args["occupancy_roi"] = 0.5

    data_cfg: MotionDatasetConfig = get_dataset_config(exp_cfg)

    model = load_model(exp_cfg)

    if no_overrides:
        return model, get_dataloader(data_cfg, split)

    data_cfg.filter_future = True
    data_cfg.random_heatmap_count = 0
    data_cfg.random_heatmap_piecewise.clear()  # Ensure piecewise random is cleared

    if isinstance(data_cfg, WaymoDatasetConfig):
        data_cfg.scenario_id = True
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

    return model.eval(), get_dataloader(data_cfg, split)
