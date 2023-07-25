"""Dataset for the INTERACTION dataset that consists of folders of CSVs
"""
from dataclasses import dataclass, asdict, field
from pathlib import Path
from subprocess import run
from typing import Any, Dict, List, Tuple

import numpy as np
from konductor.data import DATASET_REGISTRY, DatasetConfig, Mode
from nvidia.dali import pipeline_def, Pipeline, newaxis
from nvidia.dali.types import DALIDataType, Constant
import nvidia.dali.math as dmath
import nvidia.dali.fn as fn
import nvidia.dali.tfrecord as tfrec

try:
    from .utils import get_cache_record_idx_path
except ImportError:
    from utils import get_cache_record_idx_path

# fmt: off
_ALL_FEATURES = [
    "x", "y", "bbox_yaw",
    "velocity_x", "velocity_y", "vel_yaw",
    "width", "length", "class"
]
# fmt: on

_MAX_AGENTS: int = 64
_MAX_ROADGRAPH: int = 1024
_TIMESPAN: int = 40


@dataclass
@DATASET_REGISTRY.register_module("interacton")
class InteractionConfig(DatasetConfig):
    occupancy_size: int = field(kw_only=True)
    full_sequence: bool = False
    # Vehicle_features is order sensitive (this is ordering of channel concatenation)
    vehicle_features: List[str] = field(default_factory=lambda: _ALL_FEATURES)
    road_features: bool = False
    roadmap: bool = False
    roadmap_size: int | None = None
    only_vehicles: bool = False
    signal_features: bool = False
    map_normalize: float = 0.0
    heatmap_time: List[int] | None = None
    filter_future: bool = True
    separate_classes: bool = False
    random_heatmap_minmax: Tuple[int, int] | None = None
    random_heatmap_count: int = 0
    # How to scale the occupancy roi, whole image => 1, center crop => 0.5
    occupancy_roi: float = 1.0
    flow_mask: bool = False
    velocity_norm: float = 1.0
    time_stride: int = 1
    random_start: bool = False

    @property
    def properties(self) -> Dict[str, Any]:
        return asdict(self)

    def get_instance(self, mode: Mode, **kwargs) -> Tuple[Pipeline, List[str], str]:
        assert not self.signal_features, "Signal features unavailable for interaction"
        tfrec_file = self.basepath / f"interaction_{mode.name}.tfrecord"

        output_map = ["agents", "agents_valid"]
        if self.roadmap:
            output_map.append("roadmap")
        if self.occupancy_size > 0:
            output_map.extend(["time_idx", "heatmap"])
        if self.flow_mask:
            output_map.append("flow")

        datapipe = interation_pipeline(tfrec_file, cfg=self, **kwargs)
        return datapipe, output_map, tfrec_file.stem, -1


@pipeline_def
def interation_pipeline(
    record_file: Path,
    shard_id: int,
    num_shards: int,
    random_shuffle: bool,
    cfg: InteractionConfig,
    augmentations: Dict[str, Any],
):
    # fmt: off
    # Features of the road.
    roadgraph_features = {
        "roadgraph/id": tfrec.FixedLenFeature([_MAX_ROADGRAPH, 1], tfrec.int64, 0),
        "roadgraph/type": tfrec.FixedLenFeature([_MAX_ROADGRAPH, 1], tfrec.int64, 0),
        "roadgraph/valid": tfrec.FixedLenFeature([_MAX_ROADGRAPH, 1], tfrec.int64, 0),
        "roadgraph/xyz": tfrec.FixedLenFeature([_MAX_ROADGRAPH, 3], tfrec.float32, 0.0),
    }
    
    # Features of other agents.
    state_features = {
        "state/x": tfrec.FixedLenFeature([_MAX_AGENTS, _TIMESPAN], tfrec.float32, 0.0),
        "state/y": tfrec.FixedLenFeature([_MAX_AGENTS, _TIMESPAN], tfrec.float32, 0.0),
        "state/t": tfrec.FixedLenFeature([_MAX_AGENTS, _TIMESPAN], tfrec.float32, 0.0),
        "state/vx": tfrec.FixedLenFeature([_MAX_AGENTS, _TIMESPAN], tfrec.float32, 0.0),
        "state/vy": tfrec.FixedLenFeature([_MAX_AGENTS, _TIMESPAN], tfrec.float32, 0.0),
        "state/length": tfrec.FixedLenFeature([_MAX_AGENTS, _TIMESPAN], tfrec.float32, 0.0),
        "state/width": tfrec.FixedLenFeature([_MAX_AGENTS, _TIMESPAN], tfrec.float32, 0.0),
        "state/timestamp_ms": tfrec.FixedLenFeature([_MAX_AGENTS, _TIMESPAN], tfrec.int64, 0),
        "state/valid": tfrec.FixedLenFeature([_MAX_AGENTS, _TIMESPAN], tfrec.int64, 0),
        "state/id": tfrec.FixedLenFeature([_MAX_AGENTS, 1], tfrec.int64, 0),
        "state/type": tfrec.FixedLenFeature([_MAX_AGENTS], tfrec.int64, 0),
    }
    # fmt: on

    features_description = {}
    features_description.update(roadgraph_features)
    features_description.update(state_features)

    tfrec_idx_root = get_cache_record_idx_path(record_file.parent)
    tfrec_idx = tfrec_idx_root / f"{record_file.name}.idx"

    if not tfrec_idx.exists():
        run(["tfrecord2idx", str(record_file), str(tfrec_idx)])

    inputs = fn.readers.tfrecord(
        path=str(record_file),
        index_path=str(tfrec_idx),
        features=features_description,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=random_shuffle,
        name=record_file.stem,
    )

    data_valid = fn.cast(inputs[f"state/valid"], dtype=DALIDataType.INT32)

    data_xy = fn.stack(inputs["state/x"], inputs["state/y"], axis=2)
    # Center coordinate system based off vehicles
    center = fn.transforms.translation(
        offset=-fn.cat(
            fn.masked_median(inputs["state/x"], data_valid),
            fn.masked_median(inputs["state/y"], data_valid),
        )
    )
    if "random_rotate" in augmentations:
        rot_mat = fn.transforms.rotation(
            angle=fn.random_uniform(range=[-180, 180], dtype=DALIDataType.FLOAT)
        )
    else:
        rot_mat = fn.transforms.rotation(angle=0)

    xy_tf = fn.transforms.combine(center, rot_mat)

    # Transform XY
    data_xy = fn.coord_transform(fn.reshape(data_xy, shape=[-1, 2]), MT=xy_tf)
    data_xy = fn.reshape(data_xy, shape=[_MAX_AGENTS, -1, 2])

    # Transform V{X|Y}
    data_vxvy = fn.stack(inputs["state/vx"], inputs["state/vy"], axis=2)
    if cfg.velocity_norm != 1.0:
        inv_norm = 1 / cfg.velocity_norm
        vxvy_tf = fn.transforms.combine(
            rot_mat, np.array([[inv_norm, 0, 0], [0, inv_norm, 0]], np.float32)
        )
    else:
        vxvy_tf = rot_mat
    data_vxvy = fn.coord_transform(fn.reshape(data_vxvy, shape=[-1, 2]), MT=vxvy_tf)
    data_vxvy = fn.reshape(data_vxvy, shape=[_MAX_AGENTS, -1, 2])

    # Handle WL
    data_wl = fn.stack(inputs["state/width"], inputs["state/length"], axis=2)

    # Center the map at 0,0 and divide by normalization factor
    if cfg.map_normalize > 0.0:
        data_xy /= cfg.map_normalize
        data_wl /= cfg.map_normalize

        data_valid *= (dmath.abs(data_xy[:, :, 0]) < 1) * (
            dmath.abs(data_xy[:, :, 1]) < 1
        )

    # Normalize angle between [-1,1]
    data_yaw = inputs["state/t"][:, :, newaxis]
    data_yaw = dmath.atan2(dmath.sin(data_yaw), dmath.cos(data_yaw)) / Constant(
        np.pi, dtype=DALIDataType.FLOAT
    )

    # Create yaw rate, will have some broken parts when wrapping -pi -> +pi
    data_vt = data_yaw[:, 1:] - data_yaw[:, :-1]
    data_vt = fn.cat(data_vt, data_vt[:, -1:], axis=1)

    # Create class vector
    data_class = fn.cast(inputs["state/type"], dtype=DALIDataType.FLOAT)
    data_class = fn.stack(*[data_class] * _TIMESPAN if cfg.full_sequence else 1, axis=1)
    if cfg.only_vehicles:
        data_valid *= data_class == 1
    data_class = data_class[:, :, newaxis]

    data_out = [data_xy]
    if "bbox_yaw" in cfg.vehicle_features:
        data_out.append(data_yaw)
    if any("velocity" in k for k in cfg.vehicle_features):
        data_out.append(data_vxvy)
    if "vel_yaw" in cfg.vehicle_features:
        data_out.append(data_vt)
    if all(k in cfg.vehicle_features for k in ["width", "length"]):
        data_out.append(data_wl)
    if "class" in cfg.vehicle_features:
        data_out.append(data_class)

    outputs = [fn.cat(*data_out, axis=2), inputs["state/valid"]]

    if cfg.time_stride > 1:
        outputs = fn.stride_slice(outputs, axis=1, stride=cfg.time_stride)

    if cfg.roadmap:
        outputs.append(
            fn.roadgraph_image(
                inputs["roadgraph/xyz"],
                inputs["roadgraph/type"],
                inputs["roadgraph/id"],
                inputs["roadgraph/valid"],
                xy_tf,
                size=cfg.roadmap_size,
                normalize_value=cfg.map_normalize,
                lane_markings=True,
                # lane_center=True,
            )
        )

    # Add occupancy heatmap
    if cfg.occupancy_size > 0:
        rand_kwargs = {"n_random": cfg.random_heatmap_count}
        if cfg.random_heatmap_count > 0:
            assert cfg.random_heatmap_minmax is not None
            rand_kwargs["min"] = cfg.random_heatmap_minmax[0]
            rand_kwargs["max"] = cfg.random_heatmap_minmax[1]

        time_idx = fn.mixed_random_generator(
            always_sample=cfg.heatmap_time, **rand_kwargs
        )

        # Concat all features
        data_all = fn.cat(
            data_xy, data_yaw, data_vxvy, data_vt, data_wl, data_class, axis=2
        )

        if cfg.time_stride > 1:
            data_all = fn.stride_slice(data_all, axis=1, stride=cfg.time_stride)

        occ_kwargs = {
            "size": cfg.occupancy_size,
            "roi": cfg.occupancy_roi,
            "filter_future": cfg.filter_future,
            "separate_classes": cfg.separate_classes,
        }

        outputs.append(time_idx)
        outputs.append(fn.occupancy_mask(data_all, data_valid, time_idx, **occ_kwargs))
        if cfg.flow_mask:
            outputs.append(fn.flow_mask(data_all, data_valid, time_idx, **occ_kwargs))

    return tuple([o.gpu() for o in outputs])
