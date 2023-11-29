"""Dataset for the INTERACTION dataset that consists of folders of CSVs
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import nvidia.dali.math as dmath
import nvidia.dali.tfrecord as tfrec
from konductor.data import DATASET_REGISTRY, Split, ModuleInitConfig
from nvidia.dali import fn, newaxis, pipeline_def
from nvidia.dali.types import Constant, DALIDataType

try:
    from .common import (
        VALID_AUG,
        MotionDatasetConfig,
        dali_rad2deg,
        get_sample_idxs,
        get_tfrecord_cache,
    )
except ImportError:
    from common import (
        VALID_AUG,
        MotionDatasetConfig,
        dali_rad2deg,
        get_sample_idxs,
        get_tfrecord_cache,
    )


_MAX_AGENTS: int = 64
_MAX_ROADGRAPH: int = 1024
_TIMESPAN: int = 40


@dataclass
@DATASET_REGISTRY.register_module("interaction")
class InteractionConfig(MotionDatasetConfig):
    max_agents = _MAX_AGENTS
    sequence_length = _TIMESPAN
    current_time_idx = 10

    def __post_init__(self):
        super().__post_init__()
        self.basepath = self.basepath / "interaction" / "multi" / "tfrecord"

    @property
    def properties(self) -> Dict[str, Any]:
        return asdict(self)

    def get_dataloader(self, split: Split):
        tfrec_file = self.basepath / f"interaction_{split.name.lower()}.tfrecord"

        output_map = ["agents", "agents_valid"]
        if self.roadmap:
            output_map.append("roadmap")
        if self.occupancy_size > 0:
            output_map.extend(["time_idx", "heatmap"])
        if self.flow_mask:
            output_map.append("flow")
        if self.scenario_id:
            output_map.append("scenario_id")

        loader = self.train_loader if split is Split.TRAIN else self.val_loader
        datapipe = interation_pipeline(tfrec_file, cfg=self, **loader.pipe_kwargs())
        return loader.get_instance(datapipe, output_map, reader_name=tfrec_file.stem)


@pipeline_def
def interation_pipeline(
    record_file: Path,
    shard_id: int,
    num_shards: int,
    random_shuffle: bool,
    cfg: InteractionConfig,
    augmentations: List[ModuleInitConfig],
):
    assert all(a.type in VALID_AUG for a in augmentations)
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
    features_description["scenario_id"] = tfrec.FixedLenFeature([], tfrec.string, "")

    tfrec_idx = get_tfrecord_cache(record_file.parent, [record_file.name])[0]

    inputs = fn.readers.tfrecord(
        path=str(record_file),
        index_path=tfrec_idx,
        features=features_description,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=random_shuffle,
        name=record_file.stem,
    )

    data_valid = fn.cast(inputs["state/valid"], dtype=DALIDataType.INT32)

    data_xy = fn.stack(inputs["state/x"], inputs["state/y"], axis=2)
    # Center coordinate system based off median vehicle position
    center = fn.transforms.translation(
        offset=-fn.cat(
            fn.masked_median(inputs["state/x"], data_valid),
            fn.masked_median(inputs["state/y"], data_valid),
        )
    )
    if any(a.type == "random_rotate" for a in augmentations):
        angle_rad = fn.random.uniform(range=[-np.pi, np.pi], dtype=DALIDataType.FLOAT)
    else:
        angle_rad = Constant(0.0, device="cpu", dtype=DALIDataType.FLOAT)

    rot_mat = fn.transforms.rotation(
        angle=fn.reshape(dali_rad2deg(angle_rad), shape=[])
    )
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
    data_yaw = inputs["state/t"][:, :, newaxis] + angle_rad
    data_yaw = dmath.atan2(dmath.sin(data_yaw), dmath.cos(data_yaw)) / Constant(
        np.pi, dtype=DALIDataType.FLOAT
    )

    # Create yaw rate, will have some broken parts when wrapping -pi -> +pi
    data_vt = data_yaw[:, 1:] - data_yaw[:, :-1]
    data_vt = fn.cat(data_vt, data_vt[:, -1:], axis=1)

    # Create class vector
    data_class = fn.cast(inputs["state/type"], dtype=DALIDataType.FLOAT)
    data_class = fn.stack(*[data_class] * _TIMESPAN if cfg.full_sequence else 1, axis=1)
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
        time_idx = get_sample_idxs(cfg)

        # Concat all features
        data_all = fn.cat(
            data_xy, data_yaw, data_vxvy, data_vt, data_wl, data_class, axis=2
        )

        if cfg.time_stride > 1:
            data_all = fn.stride_slice(data_all, axis=1, stride=cfg.time_stride)

        occ_kwargs = {
            "size": cfg.occupancy_size,
            "roi": cfg.occupancy_roi,
            "filter_timestep": cfg.current_time_idx if cfg.filter_future else -1,
            "separate_classes": False,
        }

        outputs.append(time_idx)
        outputs.append(fn.occupancy_mask(data_all, data_valid, time_idx, **occ_kwargs))
        if cfg.flow_mask:
            outputs.append(
                fn.flow_mask(
                    data_all,
                    data_valid,
                    time_idx,
                    flow_type=cfg.flow_type,
                    **occ_kwargs,
                )
            )

    outputs = [o.gpu() for o in outputs]

    if cfg.scenario_id:
        outputs.append(fn.pad(inputs["scenario_id"], fill_value=0))

    return tuple(outputs)
