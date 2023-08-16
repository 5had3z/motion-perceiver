"""
Conglomoration of ETH and UCY pedestrain datasets into a set of tfrecords.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from konductor.data import DATASET_REGISTRY, Mode, ModuleInitConfig
from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.tfrecord as tfrec

try:
    from .common import (
        MotionDatasetConfig,
        get_cache_record_idx_path,
        get_sample_idxs,
        dali_rad2deg,
        VALID_AUG,
    )
except ImportError:
    from common import (
        MotionDatasetConfig,
        get_cache_record_idx_path,
        get_sample_idxs,
        dali_rad2deg,
        VALID_AUG,
    )

PAST_FRAMES = 8
FUTURE_FRAMES = 12
SEQUENCE_LENGTH = PAST_FRAMES + FUTURE_FRAMES
MAX_AGENTS = 83
SUBSETS = {"eth", "hotel", "uni", "zara1", "zara2", "zara3", "students1", "students3"}


@dataclass
@DATASET_REGISTRY.register_module("pedestrian")
class PedestrianDatasetConfig(MotionDatasetConfig):
    withheld: str = field(kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        assert self.withheld in SUBSETS

    @property
    def properties(self) -> Dict[str, Any]:
        return asdict(self)

    def get_instance(self, mode: Mode, **kwargs) -> Tuple[Pipeline, List[str], str]:
        tfrecords = (
            [s for s in SUBSETS if s != self.withheld]
            if mode == Mode.train
            else self.withheld
        )

        output_map = ["agents", "agents_valid"]
        if self.roadmap:
            output_map.append("context")
        if self.scenario_id:
            output_map.append("scenario_id")
        if self.occupancy_size:
            output_map.append("occupancy")

        pipeline = pedestrian_pipe(
            self.basepath, **kwargs, cfg=self, tfrecords=tfrecords
        )
        return pipeline, output_map, mode.name, -1


@pipeline_def
def pedestrian_pipe(
    record_root: Path,
    shard_id: int,
    num_shards: int,
    random_shuffle: bool,
    cfg: PedestrianDatasetConfig,
    tfrecords: List[str],
    augmentations: List[ModuleInitConfig],
):
    state_features = {
        "x": tfrec.FixedLenFeature([MAX_AGENTS, SEQUENCE_LENGTH], tfrec.float32, 0.0),
        "y": tfrec.FixedLenFeature([MAX_AGENTS, SEQUENCE_LENGTH], tfrec.float32, 0.0),
        "t": tfrec.FixedLenFeature([MAX_AGENTS, SEQUENCE_LENGTH], tfrec.float32, 0.0),
        "vx": tfrec.FixedLenFeature([MAX_AGENTS, SEQUENCE_LENGTH], tfrec.float32, 0.0),
        "vy": tfrec.FixedLenFeature([MAX_AGENTS, SEQUENCE_LENGTH], tfrec.float32, 0.0),
        "vt": tfrec.FixedLenFeature([MAX_AGENTS, SEQUENCE_LENGTH], tfrec.float32, 0.0),
        "valid": tfrec.FixedLenFeature([MAX_AGENTS, SEQUENCE_LENGTH], tfrec.int64, 0.0),
        "scenario_id": tfrec.FixedlenFeature([], tfrec.string, ""),
    }
