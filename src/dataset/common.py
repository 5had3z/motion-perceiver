from copy import deepcopy
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Tuple
from pathlib import Path
from math import pi
from subprocess import run
from multiprocessing import Pool, cpu_count
import logging

from torch.distributed import scatter_object_list
from nvidia.dali import fn
from nvidia.dali.data_node import DataNode
from nvidia.dali.types import DALIDataType, Constant
from konductor.data import DatasetConfig
from konductor.data.dali import DaliLoaderConfig
from konductor.utilities.comm import (
    get_local_rank,
    synchronize,
    in_distributed_mode,
    get_world_size,
)

# fmt: off
ALL_FEATURES = [
    "x", "y", "bbox_yaw",
    "velocity_x", "velocity_y", "vel_yaw",
    "width", "length", "class"
]
# fmt: on

VALID_AUG = {"random_rotate"}


@dataclass
class MotionDatasetConfig(DatasetConfig):
    # Motion datasets use DALI
    train_loader: DaliLoaderConfig
    val_loader: DaliLoaderConfig

    # These should be set by specific dataset
    max_agents: int = field(init=False)
    sequence_length: int = field(init=False)
    current_time_idx: int = field(init=False)

    occupancy_size: int = field(kw_only=True)
    full_sequence: bool = False
    # Vehicle_features is order sensitive (this is ordering of channel concatenation)
    vehicle_features: List[str] = field(default_factory=lambda: ALL_FEATURES)
    roadmap: bool = False
    map_normalize: float = 0.0
    heatmap_time: List[int] = field(kw_only=True)
    filter_future: bool = False
    random_heatmap_minmax: Tuple[int, int] | None = None
    random_heatmap_count: int = 0
    random_heatmap_stride: int = 1
    random_heatmap_piecewise: List[Dict[str, int]] = field(default_factory=list)
    roadmap_size: int = 0
    # How to scale the occupancy roi, whole image => 1, center crop => 0.5
    occupancy_roi: float = 1.0
    flow_mask: bool = False
    velocity_norm: float = 1.0
    time_stride: int = 1
    random_start: bool = False
    scenario_id: bool = False

    @property
    def properties(self) -> Dict[str, Any]:
        return asdict(self)

    def __post_init__(self):
        if self.roadmap_size == 0:
            self.roadmap_size = self.occupancy_size


def get_cache_record_idx_path(dataset_path: Path) -> Path:
    """
    Initially try to make with tf record dali index
    in folder adjacent to dataset suffixed by idx.
    If that fails due to permission requirements, make in /tmp.
    Only local rank 0 handles the creation logic, the result is shared
    with the rest of the nodes.
    """
    dali_idx_path = dataset_path.parent / f"{dataset_path.name}_dali_idx"

    if not dali_idx_path.exists() and get_local_rank() == 0:
        try:
            dali_idx_path.mkdir()
            return dali_idx_path
        except OSError:
            print(
                f"Unable to create dali index at {dali_idx_path},"
                f" changing to /tmp/{dataset_path.name}_dali_idx"
            )

            dali_idx_path = Path(f"/tmp/{dataset_path.name}_dali_idx")
            if not dali_idx_path.exists():
                dali_idx_path.mkdir()

    # The result from local rank 0 should be the same on each
    # node, so it should be fine if global rank 0 broadcasts
    if in_distributed_mode():
        tmp = [None]
        scatter_object_list(tmp, [dali_idx_path] * get_world_size())
        dali_idx_path: Path = tmp[0]

    return dali_idx_path


def get_tfrecord_cache(record_root: Path, tfrecords: List[str]) -> List[str]:
    """Get the list of paths of tfrecord indexes
    and create the index if necessary"""
    # Get the path to the record index folder
    tfrec_idx_root = get_cache_record_idx_path(record_root)

    # Create the path to each index
    tfrec_idxs = [tfrec_idx_root / f"{tfrec}.idx" for tfrec in tfrecords]

    # Check if index exists, write if necessary
    proc_args = []
    for tfrec, idx in zip(tfrecords, tfrec_idxs):
        if not idx.exists():
            proc_args.append(["tfrecord2idx", str(record_root / tfrec), str(idx)])

    if len(proc_args) > 0 and get_local_rank() == 0:
        logging.info("Creating %d DALI TFRecord Indexes", len(proc_args))
        with Pool(processes=cpu_count() // 2) as mp:
            mp.map(run, proc_args)

    # Ensure distributed-parallel sync
    synchronize()

    return [str(f) for f in tfrec_idxs]


def get_sample_idxs(cfg: MotionDatasetConfig):
    """Create the time idxs which the heatmap is randomly sampled from"""
    if len(cfg.random_heatmap_piecewise) == 0:
        rand_kwargs = {
            "n_random": cfg.random_heatmap_count,
            "stride": cfg.random_heatmap_stride,
        }
        if cfg.random_heatmap_count > 0:
            assert cfg.random_heatmap_minmax is not None
            rand_kwargs["min"] = cfg.random_heatmap_minmax[0]
            rand_kwargs["max"] = cfg.random_heatmap_minmax[1]
        if in_distributed_mode():
            rand_kwargs["seed"] = 12345

        time_idx = fn.mixed_random_generator(
            always_sample=cfg.heatmap_time, **rand_kwargs
        )
    else:
        # run input validation, piecewise should be in ascending order
        for before, after in zip(
            cfg.random_heatmap_piecewise, cfg.random_heatmap_piecewise[1:]
        ):
            assert before["max"] < after["min"]

        time_idxs = []
        for kwargs in cfg.random_heatmap_piecewise:
            kwargs = deepcopy(kwargs)
            if in_distributed_mode():
                kwargs["seed"] = 12345
            const_times = sorted(
                [x for x in cfg.heatmap_time if kwargs["min"] <= x < kwargs["max"]]
            )
            time_idxs.append(
                fn.mixed_random_generator(always_sample=const_times, **kwargs)
            )
        time_idx = fn.cat(*time_idxs, axis=0)
    return time_idx


def dali_rad2deg(radians: DataNode) -> DataNode:
    """Convert radians to degrees 180 * rad / pi"""
    return (
        Constant(180.0, dtype=DALIDataType.FLOAT)
        * radians
        / Constant(pi, dtype=DALIDataType.FLOAT)
    )
