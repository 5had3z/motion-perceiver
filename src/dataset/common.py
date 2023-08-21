from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Tuple
from pathlib import Path
from math import pi

from nvidia.dali import fn
from nvidia.dali.data_node import DataNode
from nvidia.dali.types import DALIDataType, Constant
from konductor.data import DatasetConfig

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
    roadmap_size: int | None = None
    # How to scale the occupancy roi, whole image => 1, center crop => 0.5
    occupancy_roi: float = 1.0
    flow_mask: bool = False
    velocity_norm: float = 1.0
    time_stride: int = 1
    random_start: bool = False
    scenario_id: bool = False

    @property
    def properties(self) -> Dict[str, Any]:
        props = asdict(self)
        return props

    def __post_init__(self):
        if self.roadmap_size is None:
            self.roadmap_size = self.occupancy_size


def get_cache_record_idx_path(dataset_path: Path) -> Path:
    """
    Initially try to make with tf record dali index
    in folder adjacent to dataset suffixed by idx.
    If that fails due to permission requirements, make in /tmp.
    """
    dali_idx_path = dataset_path.parent / f"{dataset_path.name}_dali_idx"
    if not dali_idx_path.exists():
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

    return dali_idx_path


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
