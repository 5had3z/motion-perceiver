"""
Conglomoration of ETH and UCY pedestrain datasets into a set of tfrecords.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from subprocess import run
from typing import Any, Dict, List, Tuple

import numpy as np
from konductor.data import DATASET_REGISTRY, Mode, ModuleInitConfig
from nvidia.dali import pipeline_def, Pipeline, fn, newaxis
from nvidia.dali.types import Constant, DALIDataType
import nvidia.dali.tfrecord as tfrec
import nvidia.dali.math as dmath

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
            else [self.withheld]
        )
        tfrecords = [t + ".tfrecord" for t in tfrecords]

        output_map = ["agents", "agents_valid"]
        if self.roadmap:
            output_map.append("context")
        if self.occupancy_size:
            output_map.extend(["time_idx", "occupancy"])
        if self.scenario_id:
            output_map.append("scenario_id")

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
    rec_features = {
        "x": tfrec.FixedLenFeature([MAX_AGENTS, SEQUENCE_LENGTH], tfrec.float32, 0.0),
        "y": tfrec.FixedLenFeature([MAX_AGENTS, SEQUENCE_LENGTH], tfrec.float32, 0.0),
        "t": tfrec.FixedLenFeature([MAX_AGENTS, SEQUENCE_LENGTH], tfrec.float32, 0.0),
        "vx": tfrec.FixedLenFeature([MAX_AGENTS, SEQUENCE_LENGTH], tfrec.float32, 0.0),
        "vy": tfrec.FixedLenFeature([MAX_AGENTS, SEQUENCE_LENGTH], tfrec.float32, 0.0),
        "vt": tfrec.FixedLenFeature([MAX_AGENTS, SEQUENCE_LENGTH], tfrec.float32, 0.0),
        "valid": tfrec.FixedLenFeature([MAX_AGENTS, SEQUENCE_LENGTH], tfrec.int64, 0),
        "scenario_id": tfrec.FixedLenFeature([], tfrec.string, ""),
    }

    tfrec_idx_root = get_cache_record_idx_path(record_root)

    def record_idx(tf_record: str) -> Path:
        return tfrec_idx_root / f"{tf_record}.idx"

    for fragment in tfrecords:
        tfrec_idx = record_idx(fragment)
        if not tfrec_idx.exists():
            run(["tfrecord2idx", str(record_root / fragment), str(tfrec_idx)])

    inputs = fn.readers.tfrecord(
        path=[str(record_root / r) for r in tfrecords],
        index_path=[str(record_idx(r)) for r in tfrecords],
        features=rec_features,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=random_shuffle,
        name="train" if len(tfrecords) > 1 else "val",
    )

    data_valid = fn.cast(inputs["valid"], dtype=DALIDataType.INT32)

    if any(a.type == "random_rotate" for a in augmentations):
        angle_rad = fn.random.uniform(range=[-np.pi, np.pi], dtype=DALIDataType.FLOAT)
    else:
        angle_rad = Constant(0.0, dtype=DALIDataType.FLOAT)
    rot_mat = fn.transforms.rotation(angle=dali_rad2deg(angle_rad))

    data_xy = fn.stack(inputs["x"], inputs["y"], axis=2)
    data_xy = fn.coord_transform(fn.reshape(data_xy, shape=[-1, 2]), MT=rot_mat)
    data_xy = fn.reshape(data_xy, shape=[MAX_AGENTS, SEQUENCE_LENGTH, 2])

    data_vxvy = fn.stack(inputs["vx"], inputs["vy"], axis=2)
    if cfg.velocity_norm != 1.0:
        inv_norm = 1.0 / cfg.velocity_norm
        vxvy_tf = fn.transforms.combine(
            rot_mat, np.array([[inv_norm, 0, 0], [0, inv_norm, 0]], np.float32)
        )
    else:
        vxvy_tf = rot_mat
    data_vxvy = fn.coord_transform(fn.reshape(data_vxvy, shape=[-1, 2]), MT=vxvy_tf)
    data_vxvy = fn.reshape(data_vxvy, shape=[MAX_AGENTS, SEQUENCE_LENGTH, 2])

    if cfg.map_normalize > 0.0:
        data_xy /= cfg.map_normalize
        data_valid *= (dmath.abs(data_xy[:, :, 0]) < 1) * (
            dmath.abs(data_xy[:, :, 1]) < 1
        )

    data_yaw = inputs["t"][:, :, newaxis] + angle_rad
    data_yaw = dmath.atan2(dmath.sin(data_yaw), dmath.cos(data_yaw)) / Constant(
        np.pi, dtype=DALIDataType.FLOAT
    )

    data_vt = inputs["vt"][:, :, newaxis]

    data_out = [data_xy]
    if "bbox_yaw" in cfg.vehicle_features:
        data_out.append(data_yaw)
    if any("velocity" in k for k in cfg.vehicle_features):
        data_out.append(data_vxvy)
    if "vel_yaw" in cfg.vehicle_features:
        data_out.append(data_vt)
    outputs = [fn.cat(*data_out, axis=2), inputs["valid"]]

    if cfg.time_stride > 1:
        outputs = fn.stride_slice(outputs, axis=1, stride=cfg.time_stride)

    if cfg.roadmap:
        raise NotImplementedError()

    if cfg.occupancy_size > 0:
        time_idx = get_sample_idxs(cfg)
        data_all = fn.cat(data_xy, data_yaw, data_vxvy, data_vt, axis=2)

        if cfg.time_stride > 1:
            data_all = fn.stride_slice(data_all, axis=1, stride=cfg.time_stride)

        occ_kwargs = {
            "size": cfg.occupancy_size,
            "roi": cfg.occupancy_roi,
            "filter_future": cfg.filter_future,
            "separate_classes": False,
            "circle_radius": 0.5 / cfg.map_normalize,
        }

        outputs.append(time_idx)
        outputs.append(fn.occupancy_mask(data_all, data_valid, time_idx, **occ_kwargs))
        if cfg.flow_mask:
            raise NotImplementedError()

    outputs = [o.gpu() for o in outputs]

    if cfg.scenario_id:
        outputs.append(fn.pad("scenario_id", fill_value=0))

    return tuple(outputs)
