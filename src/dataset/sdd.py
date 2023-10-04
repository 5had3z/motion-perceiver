"""
Stanford drone dataset
"""
from math import ceil
from dataclasses import dataclass, field
from typing import List, Tuple

from konductor.data import DATASET_REGISTRY, Split
from nvidia.dali import Pipeline
import yaml

from .common import MotionDatasetConfig
from .pedestrain_pipe import pedestrian_pipe

# fmt: off
_ALL_FEATURES =  [
    "x", "y", "bbox_yaw",
    "velocity_x", "velocity_y",
    "vel_yaw", "class",
]
# fmt: on


@dataclass
@DATASET_REGISTRY.register_module("sdd")
class SDDDatasetConfig(MotionDatasetConfig):
    vehicle_features: List[str] = field(default_factory=lambda: _ALL_FEATURES)

    def __post_init__(self):
        self.basepath /= "sdd_tfrecord"
        with open(self.basepath / "metadata.yaml", "r", encoding="utf-8") as f:
            metadata = yaml.safe_load(f)
        assert metadata["dataset"] == "sdd"

        self.max_agents = metadata["max_agents"]
        self.sequence_length = ceil(
            (metadata["history_sec"] + metadata["future_sec"]) / metadata["period_sec"]
            + 1
        )
        self.current_time_idx = ceil(metadata["history_sec"] / metadata["period_sec"])

        return super().__post_init__()

    def get_instance(self, split: Split, **kwargs) -> Tuple[Pipeline, List[str], str]:
        tfrecords = [f"sdd_{split.name.lower()}.tfrecord"]

        output_map = ["agents", "agents_valid"]
        if self.roadmap:
            output_map.append("roadmap")
        if self.occupancy_size:
            output_map.extend(["time_idx", "heatmap"])
        if self.scenario_id:
            output_map.append("scenario_id")

        pipeline = pedestrian_pipe(
            self.basepath,
            **kwargs,
            cfg=self,
            tfrecords=tfrecords,
            split=split.name.lower(),
        )
        return pipeline, output_map, split.name.lower(), -1
