from pathlib import Path
from subprocess import run
from typing import List

import numpy as np
from konductor.init import ModuleInitConfig
from nvidia.dali import pipeline_def, fn, newaxis
from nvidia.dali.types import Constant, DALIDataType
import nvidia.dali.tfrecord as tfrec
import nvidia.dali.math as dmath

try:
    from .common import (
        MotionDatasetConfig,
        get_cache_record_idx_path,
        get_sample_idxs,
        dali_rad2deg,
    )
except ImportError:
    from common import (
        MotionDatasetConfig,
        get_cache_record_idx_path,
        get_sample_idxs,
        dali_rad2deg,
    )


@pipeline_def
def pedestrian_pipe(
    record_root: Path,
    shard_id: int,
    num_shards: int,
    random_shuffle: bool,
    cfg: MotionDatasetConfig,
    tfrecords: List[str],
    split: str,
    augmentations: List[ModuleInitConfig],
):
    rec_features = {
        "type": tfrec.FixedLenFeature([cfg.max_agents], tfrec.int64, 0),
        "x": tfrec.FixedLenFeature(
            [cfg.max_agents, cfg.sequence_length], tfrec.float32, 0.0
        ),
        "y": tfrec.FixedLenFeature(
            [cfg.max_agents, cfg.sequence_length], tfrec.float32, 0.0
        ),
        "t": tfrec.FixedLenFeature(
            [cfg.max_agents, cfg.sequence_length], tfrec.float32, 0.0
        ),
        "vx": tfrec.FixedLenFeature(
            [cfg.max_agents, cfg.sequence_length], tfrec.float32, 0.0
        ),
        "vy": tfrec.FixedLenFeature(
            [cfg.max_agents, cfg.sequence_length], tfrec.float32, 0.0
        ),
        "vt": tfrec.FixedLenFeature(
            [cfg.max_agents, cfg.sequence_length], tfrec.float32, 0.0
        ),
        "valid": tfrec.FixedLenFeature(
            [cfg.max_agents, cfg.sequence_length], tfrec.int64, 0
        ),
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
        name=split,
    )

    data_valid = fn.cast(inputs["valid"], dtype=DALIDataType.INT32)

    if any(a.type == "random_rotate" for a in augmentations):
        angle_rad = fn.random.uniform(range=[-np.pi, np.pi], dtype=DALIDataType.FLOAT)
    else:
        angle_rad = Constant(0.0, device="cpu", dtype=DALIDataType.FLOAT)
    rot_mat = fn.transforms.rotation(
        angle=fn.reshape(dali_rad2deg(angle_rad), shape=[])
    )
    if any(a.type == "center" for a in augmentations):
        center = fn.transforms.translation(
            offset=-fn.cat(
                fn.masked_median(inputs["x"], data_valid),
                fn.masked_median(inputs["y"], data_valid),
            )
        )
    else:
        center = Constant([0.0, 0.0], dtype=DALIDataType.FLOAT32)
    xy_tf = fn.transforms.combine(center, rot_mat)

    data_xy = fn.stack(inputs["x"], inputs["y"], axis=2)
    data_xy = fn.coord_transform(fn.reshape(data_xy, shape=[-1, 2]), MT=xy_tf)
    data_xy = fn.reshape(data_xy, shape=[cfg.max_agents, cfg.sequence_length, 2])

    data_vxvy = fn.stack(inputs["vx"], inputs["vy"], axis=2)
    if cfg.velocity_norm != 1.0:
        inv_norm = 1.0 / cfg.velocity_norm
        vxvy_tf = fn.transforms.combine(
            rot_mat, np.array([[inv_norm, 0, 0], [0, inv_norm, 0]], np.float32)
        )
    else:
        vxvy_tf = rot_mat
    data_vxvy = fn.coord_transform(fn.reshape(data_vxvy, shape=[-1, 2]), MT=vxvy_tf)
    data_vxvy = fn.reshape(data_vxvy, shape=[cfg.max_agents, cfg.sequence_length, 2])

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
        ctx_image = fn.load_scene(
            inputs["scenario_id"],
            xy_tf,
            src="",
            metadata="",
            size=cfg.roadmap_size,
            channels=3,
        )

    if cfg.occupancy_size > 0:
        time_idx = get_sample_idxs(cfg)
        data_all = fn.cat(data_xy, data_yaw, data_vxvy, data_vt, axis=2)

        if cfg.time_stride > 1:
            data_all = fn.stride_slice(data_all, axis=1, stride=cfg.time_stride)

        occ_kwargs = {
            "size": cfg.occupancy_size,
            "roi": cfg.occupancy_roi,
            "filter_timestep": cfg.current_time_idx if cfg.filter_future else -1,
            "separate_classes": False,
            "circle_radius": 0.5 / cfg.map_normalize,
        }

        outputs.append(time_idx)
        outputs.append(fn.occupancy_mask(data_all, data_valid, time_idx, **occ_kwargs))
        if cfg.flow_mask:
            raise NotImplementedError()

    outputs = [o.gpu() for o in outputs]

    if cfg.scenario_id:
        outputs.append(fn.pad(inputs["scenario_id"], fill_value=0))

    return tuple(outputs)
