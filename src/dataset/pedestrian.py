"""
Conglomoration of ETH and UCY pedestrain datasets into a set of tfrecords.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from konductor.data import DATASET_REGISTRY, Mode, ModuleInitConfig
from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.tfrecord as tfrec

from .common import (
    get_cache_record_idx_path,
    MotionDatasetConfig,
    get_sample_idxs,
    VALID_AUG,
    dali_rad2deg,
)

SUBSETS = {"ETH", "Hotel", "Univ", "Zara1", "Zara2"}


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
        return super().get_instance(mode, **kwargs)


@pipeline_def
def pedestrian_pipe(
    record_root: Path,
    shard_id: int,
    num_shards: int,
    random_shuffle: bool,
    cfg: PedestrianDatasetConfig,
    augmentations: List[ModuleInitConfig],
):
    state_features = {"state/x": tfrec.FixedLenFeature([])}
