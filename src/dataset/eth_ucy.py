"""
Conglomoration of ETH and UCY pedestrain datasets into a set of tfrecords.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import yaml
from konductor.data import DATASET_REGISTRY, Split

from .common import MotionDatasetConfig
from .pedestrain_pipe import pedestrian_pipe


@dataclass
@DATASET_REGISTRY.register_module("eth-ucy")
class ETHUCYDatasetConfig(MotionDatasetConfig):
    subsets: list[str] = field(init=False)
    withheld: str = field(kw_only=True)
    current_time_idx: int = 8

    def __post_init__(self):
        super().__post_init__()
        with open(self.basepath / "metadata.yaml", "r", encoding="utf-8") as f:
            metadata = yaml.safe_load(f)
        self.subsets = metadata["subsets"]
        assert self.withheld in self.subsets
        self.sequence_length = metadata["sequence_length"]
        self.max_agents = metadata["max_agents"]

    @property
    def properties(self) -> Dict[str, Any]:
        return asdict(self)

    def get_dataloader(self, split: Split):
        tfrecords = (
            [s for s in self.subsets if s != self.withheld]
            if split is Split.TRAIN
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

        loader = self.train_loader if split is Split.TRAIN else self.val_loader
        pipeline = pedestrian_pipe(
            self.basepath,
            cfg=self,
            tfrecords=tfrecords,
            split=split.name.lower(),
            **loader.pipe_kwargs()
        )
        return loader.get_instance(pipeline, output_map, reader_name=split.name.lower())
