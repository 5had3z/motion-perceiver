"""
Conglomoration of ETH and UCY pedestrain datasets into a set of tfrecords.
"""

from dataclasses import dataclass, asdict, field

from typing import Any, Dict, List, Tuple

from konductor.data import DATASET_REGISTRY, Split
from nvidia.dali import Pipeline

try:
    from .common import MotionDatasetConfig
    from .pedestrain_pipe import pedestrian_pipe
except ImportError:
    from common import MotionDatasetConfig

PAST_FRAMES = 8
FUTURE_FRAMES = 12
SEQUENCE_LENGTH = PAST_FRAMES + FUTURE_FRAMES
MAX_AGENTS = 83
SUBSETS = {"eth", "hotel", "uni", "zara1", "zara2", "zara3", "students1", "students3"}


@dataclass
@DATASET_REGISTRY.register_module("eth-ucy")
class ETHUCYDatasetConfig(MotionDatasetConfig):
    withheld: str = field(kw_only=True)
    sequence_length: int = SEQUENCE_LENGTH
    max_agents: int = MAX_AGENTS
    current_time_idx: int = PAST_FRAMES

    def __post_init__(self):
        super().__post_init__()
        assert self.withheld in SUBSETS
        self.basepath /= "eth_ucy_tfrecord"

    @property
    def properties(self) -> Dict[str, Any]:
        return asdict(self)

    def get_instance(self, mode: Split, **kwargs) -> Tuple[Pipeline, List[str], str]:
        tfrecords = (
            [s for s in SUBSETS if s != self.withheld]
            if mode == Split.TRAIN
            else [self.withheld]
        )
        tfrecords = [t + ".tfrecord" for t in tfrecords]

        output_map = ["agents", "agents_valid"]
        if self.roadmap:
            output_map.append("roadmap")
        if self.occupancy_size:
            output_map.extend(["time_idx", "heatmap"])
        if self.scenario_id:
            output_map.append("scenario_id")

        pipeline = pedestrian_pipe(
            self.basepath, **kwargs, cfg=self, tfrecords=tfrecords, split=mode.name
        )
        return pipeline, output_map, mode.name, -1
