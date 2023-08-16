from dataclasses import dataclass, asdict
from subprocess import run
from multiprocessing import Pool, cpu_count
from pathlib import Path
import math
from typing import Any, Dict, List, Tuple

import numpy as np
from nvidia.dali import pipeline_def, Pipeline, newaxis, fn
from nvidia.dali.types import DALIDataType, Constant
import nvidia.dali.math as dmath
import nvidia.dali.tfrecord as tfrec
from konductor.data import DATASET_REGISTRY, Mode, ModuleInitConfig

from .common import (
    get_cache_record_idx_path,
    MotionDatasetConfig,
    get_sample_idxs,
    VALID_AUG,
    dali_rad2deg,
)


_TIME_KEYS = ["past", "current", "future"]


@dataclass
@DATASET_REGISTRY.register_module("waymo_motion")
class WaymoDatasetConfig(MotionDatasetConfig):
    signal_features: bool = False
    separate_classes: bool = False
    only_vehicles: bool = False
    scenario_id: bool = False
    use_sdc_frame: bool = False
    waymo_eval_frame: bool = False
    sdc_index: bool = False

    @property
    def properties(self) -> Dict[str, Any]:
        props = asdict(self)
        if self.separate_classes:
            props["classes"] = ["vehicles", "pedestrains", "cyclists"]
        return props

    def __post_init__(self):
        if self.heatmap_time is None:
            self.heatmap_time = [10] if self.full_sequence else [0]
        if self.roadmap_size is None:
            self.roadmap_size = self.occupancy_size

    def get_instance(self, mode: Mode, **kwargs) -> Tuple[Pipeline, List[str], str]:
        root = {
            Mode.train: self.basepath / "training",
            Mode.val: self.basepath / "validation",
            Mode.test: self.basepath / "testing",
        }[mode]

        output_map = ["agents", "agents_valid"]
        if self.signal_features:
            output_map.extend(["signals", "signals_valid"])
        if self.roadmap:
            output_map.append("roadmap")
        if self.occupancy_size > 0:
            output_map.extend(["time_idx", "heatmap"])
        if self.flow_mask:
            output_map.append("flow")
        if self.scenario_id:
            output_map.append("scenario_id")
        if self.sdc_index:
            output_map.append("sdc_mask")

        datapipe = waymo_motion_pipe(root, cfg=self, **kwargs)
        return datapipe, output_map, root.stem, -1


@pipeline_def
def waymo_motion_pipe(
    record_root: Path,
    shard_id: int,
    num_shards: int,
    random_shuffle: bool,
    cfg: WaymoDatasetConfig,
    augmentations: List[ModuleInitConfig],
):
    """Waymo data should be split in separate folders
    training/validation/testing. Therefore we should be able
    to determine the split by the folder name"""
    assert all(a.type in VALID_AUG for a in augmentations)

    # fmt: off
    # Features of the road.
    roadgraph_features = {
        # "roadgraph_samples/dir": tfrec.FixedLenFeature([20000, 3], tfrec.float32, 0.0),
        "roadgraph_samples/id": tfrec.FixedLenFeature([20000, 1], tfrec.int64, 0),
        "roadgraph_samples/type": tfrec.FixedLenFeature([20000, 1], tfrec.int64, 0),
        "roadgraph_samples/valid": tfrec.FixedLenFeature([20000, 1], tfrec.int64, 0),
        "roadgraph_samples/xyz": tfrec.FixedLenFeature([20000, 3], tfrec.float32, 0.0),
    }

    # Features of other agents.
    state_features = {
        # "state/id": tfrec.FixedLenFeature([128], tfrec.float32, 0.0),
        "state/type": tfrec.FixedLenFeature([128], tfrec.float32, 0.0),
        "state/is_sdc": tfrec.FixedLenFeature([128], tfrec.int64, 0),
        # "state/tracks_to_predict": tfrec.FixedLenFeature([128], tfrec.int64, 0),
        "state/current/bbox_yaw": tfrec.FixedLenFeature([128, 1], tfrec.float32, 0.0),
        "state/current/height": tfrec.FixedLenFeature([128, 1], tfrec.float32, 0.0),
        "state/current/length": tfrec.FixedLenFeature([128, 1], tfrec.float32, 0.0),
        # "state/current/timestamp_micros": tfrec.FixedLenFeature([128, 1], tfrec.int64, 0),
        "state/current/valid": tfrec.FixedLenFeature([128, 1], tfrec.int64, 0),
        "state/current/vel_yaw": tfrec.FixedLenFeature([128, 1], tfrec.float32, 0.0),
        "state/current/velocity_x": tfrec.FixedLenFeature([128, 1], tfrec.float32, 0.0),
        "state/current/velocity_y": tfrec.FixedLenFeature([128, 1], tfrec.float32, 0.0),
        "state/current/width": tfrec.FixedLenFeature([128, 1], tfrec.float32, 0.0),
        "state/current/x": tfrec.FixedLenFeature([128, 1], tfrec.float32, 0.0),
        "state/current/y": tfrec.FixedLenFeature([128, 1], tfrec.float32, 0.0),
        # "state/current/z": tfrec.FixedLenFeature([128, 1], tfrec.float32, 0.0),
        "state/future/bbox_yaw": tfrec.FixedLenFeature([128, 80], tfrec.float32, 0.0),
        "state/future/height": tfrec.FixedLenFeature([128, 80], tfrec.float32, 0.0),
        "state/future/length": tfrec.FixedLenFeature([128, 80], tfrec.float32, 0.0),
        # "state/future/timestamp_micros": tfrec.FixedLenFeature([128, 80], tfrec.int64, 0),
        "state/future/valid": tfrec.FixedLenFeature([128, 80], tfrec.int64, 0),
        "state/future/vel_yaw": tfrec.FixedLenFeature([128, 80], tfrec.float32, 0.0),
        "state/future/velocity_x": tfrec.FixedLenFeature([128, 80], tfrec.float32, 0.0),
        "state/future/velocity_y": tfrec.FixedLenFeature([128, 80], tfrec.float32, 0.0),
        "state/future/width": tfrec.FixedLenFeature([128, 80], tfrec.float32, 0.0),
        "state/future/x": tfrec.FixedLenFeature([128, 80], tfrec.float32, 0.0),
        "state/future/y": tfrec.FixedLenFeature([128, 80], tfrec.float32, 0.0),
        # "state/future/z": tfrec.FixedLenFeature([128, 80], tfrec.float32, 0.0),
        "state/past/bbox_yaw": tfrec.FixedLenFeature([128, 10], tfrec.float32, 0.0),
        "state/past/height": tfrec.FixedLenFeature([128, 10], tfrec.float32, 0.0),
        "state/past/length": tfrec.FixedLenFeature([128, 10], tfrec.float32, 0.0),
        # "state/past/timestamp_micros": tfrec.FixedLenFeature([128, 10], tfrec.int64, 0),
        "state/past/valid": tfrec.FixedLenFeature([128, 10], tfrec.int64, 0),
        "state/past/vel_yaw": tfrec.FixedLenFeature([128, 10], tfrec.float32, 0.0),
        "state/past/velocity_x": tfrec.FixedLenFeature([128, 10], tfrec.float32, 0.0),
        "state/past/velocity_y": tfrec.FixedLenFeature([128, 10], tfrec.float32, 0.0),
        "state/past/width": tfrec.FixedLenFeature([128, 10], tfrec.float32, 0.0),
        "state/past/x": tfrec.FixedLenFeature([128, 10], tfrec.float32, 0.0),
        "state/past/y": tfrec.FixedLenFeature([128, 10], tfrec.float32, 0.0),
        # "state/past/z": tfrec.FixedLenFeature([128, 10], tfrec.float32, 0.0),
    }

    # Features of traffic lights.
    traffic_light_features = {
        "traffic_light_state/current/state": tfrec.FixedLenFeature([1, 16], tfrec.int64, 0),
        "traffic_light_state/current/valid": tfrec.FixedLenFeature([1, 16], tfrec.int64, 0),
        "traffic_light_state/current/x": tfrec.FixedLenFeature([1, 16], tfrec.float32, 0.0),
        "traffic_light_state/current/y": tfrec.FixedLenFeature([1, 16], tfrec.float32, 0.0),
        # "traffic_light_state/current/z": tfrec.FixedLenFeature([1, 16], tfrec.float32, 0.0),
        "traffic_light_state/past/state": tfrec.FixedLenFeature([10, 16], tfrec.int64, 0),
        "traffic_light_state/past/valid": tfrec.FixedLenFeature([10, 16], tfrec.int64, 0),
        "traffic_light_state/past/x": tfrec.FixedLenFeature([10, 16], tfrec.float32, 0.0),
        "traffic_light_state/past/y": tfrec.FixedLenFeature([10, 16], tfrec.float32, 0.0),
        # "traffic_light_state/past/z": tfrec.FixedLenFeature([10, 16], tfrec.float32, 0.0),
        "traffic_light_state/future/state": tfrec.FixedLenFeature([80, 16], tfrec.int64, 0),
        "traffic_light_state/future/valid": tfrec.FixedLenFeature([80, 16], tfrec.int64, 0),
        "traffic_light_state/future/x": tfrec.FixedLenFeature([80, 16], tfrec.float32, 0.0),
        "traffic_light_state/future/y": tfrec.FixedLenFeature([80, 16], tfrec.float32, 0.0),
        # "traffic_light_state/future/z": tfrec.FixedLenFeature([80, 16], tfrec.float32, 0.0),
    }
    # fmt: on

    features_description = {}
    features_description.update(roadgraph_features)
    features_description.update(state_features)
    features_description.update(traffic_light_features)
    features_description["scenario/id"] = tfrec.FixedLenFeature([], tfrec.string, "")

    tfrec_idx_root = get_cache_record_idx_path(record_root)

    def tfrecord_idx(tf_record: Path) -> Path:
        return tfrec_idx_root / f"{tf_record.name}.idx"

    proc_args = []

    for record_fragment in record_root.iterdir():
        tfrec_idx = tfrecord_idx(record_fragment)
        if not tfrec_idx.exists():
            proc_args.append(["tfrecord2idx", str(record_fragment), str(tfrec_idx)])

    # Very IO intense (parsing 100's GB) better be NVMe
    with Pool(processes=cpu_count() // 2) as mp:
        mp.map(run, proc_args)

    inputs = fn.readers.tfrecord(
        path=[str(rec) for rec in record_root.iterdir()],
        index_path=[str(tfrecord_idx(rec)) for rec in record_root.iterdir()],
        features=features_description,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=random_shuffle,
        name=record_root.stem,
    )

    # data contains all feature information required
    _time_keys = _TIME_KEYS if cfg.full_sequence else ["current"]
    data_valid = fn.cast(
        fn.cat(*[inputs[f"state/{time_}/valid"] for time_ in _time_keys], axis=1),
        dtype=DALIDataType.INT32,
    )

    def stack_keys(*keys):
        """Stacks raw data to [inst, time, keys]"""
        assert "class" not in keys
        time_cat = [
            fn.cat(*[inputs[f"state/{time_}/{key}"] for time_ in _time_keys], axis=1)
            for key in keys
        ]
        return fn.stack(*time_cat, axis=2)

    data_xy = stack_keys("x", "y")

    if cfg.use_sdc_frame:  # Center based on SDC
        sdc_mask = inputs["state/is_sdc"] * inputs["state/current/valid"][:, 0]
        sdc_mask = fn.cast(sdc_mask, dtype=DALIDataType.INT32)
        center = fn.transforms.translation(
            offset=-fn.cat(
                fn.masked_median(inputs["state/current/x"], sdc_mask),
                fn.masked_median(inputs["state/current/y"], sdc_mask),
            )
        )
        angle_rad = Constant(math.pi / 2, dtype=DALIDataType.FLOAT) - fn.masked_median(
            inputs["state/current/bbox_yaw"], sdc_mask
        )
    else:  #  Center system based on median agent position
        center = fn.transforms.translation(
            offset=-fn.cat(
                fn.masked_median(data_xy[:, :, 0], data_valid),
                fn.masked_median(data_xy[:, :, 1], data_valid),
            )
        )
        angle_rad = Constant(0.0, dtype=DALIDataType.FLOAT)

    if any(a.type == "random_rotate" for a in augmentations):
        angle_rad += fn.random.uniform(range=[-np.pi, np.pi], dtype=DALIDataType.FLOAT)

    rot_mat = fn.transforms.rotation(
        angle=fn.reshape(dali_rad2deg(angle_rad), shape=[])
    )

    xy_tf_list = [center, rot_mat]
    if cfg.waymo_eval_frame:  # Flip y and move down 20m
        assert (
            "random_rotate" not in augmentations
        ), "Can't be in eval frame if using augmentations"
        xy_tf_list.append(np.array([[1, 0, 0], [0, -1, 20]], dtype=np.float32))
    xy_tf = fn.transforms.combine(*xy_tf_list)

    # Transform XY
    data_xy = fn.coord_transform(fn.reshape(data_xy, shape=[-1, 2]), MT=xy_tf)
    data_xy = fn.reshape(data_xy, shape=[128, -1, 2])

    # Transform V{X|Y}
    data_vxvy = stack_keys("velocity_x", "velocity_y")
    vel_tfms = []
    if cfg.waymo_eval_frame:
        vel_tfms.append(np.array([[1, 0, 0], [0, -1, 0]], np.float32))
    if cfg.velocity_norm != 1.0:
        inv_norm = 1 / cfg.velocity_norm
        vel_tfms.append(np.array([[inv_norm, 0, 0], [0, inv_norm, 0]], np.float32))
    if len(vel_tfms) > 0:
        vxvy_tf = fn.transforms.combine(rot_mat, *vel_tfms)
    else:
        vxvy_tf = rot_mat

    data_vxvy = fn.coord_transform(fn.reshape(data_vxvy, shape=[-1, 2]), MT=vxvy_tf)
    data_vxvy = fn.reshape(data_vxvy, shape=[128, -1, 2])

    data_wl = stack_keys("width", "length")

    if cfg.map_normalize > 0.0:  # Divide by normalization factor
        data_xy /= cfg.map_normalize
        data_wl /= cfg.map_normalize

        # x and y are within the ROI
        data_valid *= fn.reshape(
            (dmath.abs(data_xy[:, :, 0]) < 1) * (dmath.abs(data_xy[:, :, 1]) < 1),
            shape=[128, -1],
        )

    # normalize angle bewteen [-1,1]
    data_yaw = stack_keys("bbox_yaw")
    data_yaw += angle_rad
    data_yaw = dmath.atan2(dmath.sin(data_yaw), dmath.cos(data_yaw)) / Constant(
        math.pi, dtype=DALIDataType.FLOAT
    )
    if cfg.waymo_eval_frame:  # flip sign
        data_yaw *= -1

    # Maybe an idea for later, will stick with xy for now
    # if all(key in vehicle_features for key in ["velocity_x", "velocity_y"]):
    #     # kind of only an edge case when people don't traverse the way they're facing
    #     # and cars can only traverse the way they're facing unless they're cooked
    #     data["vel"] = dmath.sqrt(
    #         dmath.pow(data["velocity_x"], 2) + dmath.pow(data["velocity_y"], 2)
    #     )

    # Handle class stacking specially
    data_class = fn.stack(
        *[inputs["state/type"]] * 91 if cfg.full_sequence else 1, axis=1
    )[:, :, newaxis]
    if cfg.only_vehicles:  # Mark all other classes as invalid
        data_valid *= fn.reshape(data_class == 1, shape=[128, -1])

    data_vt = stack_keys("vel_yaw")

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

    outputs = [fn.cat(*data_out, axis=2), data_valid]

    # traffic signal into tensor [id, time]
    if cfg.signal_features:
        signal_valid = fn.transpose(
            fn.cast(
                fn.cat(
                    *[
                        inputs[f"traffic_light_state/{time_}/valid"]
                        for time_ in _time_keys
                    ],
                    axis=0,
                ),
                dtype=DALIDataType.INT32,
            ),
            perm=[1, 0],
        )

        signal_state = fn.transpose(
            fn.cast(
                fn.cat(
                    *[
                        inputs[f"traffic_light_state/{time_}/state"]
                        for time_ in _time_keys
                    ],
                    axis=0,
                ),
                dtype=DALIDataType.FLOAT,
            ),
            perm=[1, 0],
        )[:, :, newaxis]

        # Read signal positions over time
        signal_xy = fn.stack(
            *[
                fn.cat(
                    *[
                        inputs[f"traffic_light_state/{time_}/{k}"]
                        for time_ in _time_keys
                    ],
                    axis=0,
                )
                for k in ["x", "y"]
            ],
            axis=2,
        )

        signal_xy = fn.coord_transform(fn.reshape(signal_xy, shape=[-1, 2]), MT=xy_tf)
        signal_xy = fn.transpose(
            fn.reshape(signal_xy, shape=[-1, 16, 2]), perm=[1, 0, 2]
        )

        if cfg.map_normalize > 0.0:  # Normalize Signal Positions
            signal_xy /= cfg.map_normalize

            # Signals must be in map roi
            signal_valid *= (dmath.abs(signal_xy[:, :, 0]) < 1) * (
                dmath.abs(signal_xy[:, :, 1]) < 1
            )

        outputs.extend([fn.cat(signal_xy, signal_state, axis=2), signal_valid])

    if cfg.time_stride > 1:
        outputs = fn.stride_slice(outputs, axis=1, stride=cfg.time_stride)

    # roadmap into image
    if cfg.roadmap:
        outputs.append(
            fn.roadgraph_image(
                inputs["roadgraph_samples/xyz"],
                inputs["roadgraph_samples/type"],
                inputs["roadgraph_samples/id"],
                inputs["roadgraph_samples/valid"],
                xy_tf,
                size=cfg.roadmap_size,
                normalize_value=cfg.map_normalize,
                # lane_center=True,
                lane_markings=True,
            )
        )

    # Add occupancy heatmap
    if cfg.occupancy_size > 0:
        # Time index sample generation
        time_idx = get_sample_idxs(cfg)

        # Conat all features
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

    # Send outputs to gpu before adding cpu-only data
    outputs = [o.gpu() for o in outputs]

    if cfg.scenario_id:
        # Add padding since scenario_id contains <=16 characters
        scenario_pad = fn.pad(inputs["scenario/id"], fill_value=0)
        outputs.append(scenario_pad)

    if cfg.sdc_index:
        outputs.append(
            fn.cast(
                inputs["state/is_sdc"] * inputs["state/current/valid"][:, 0],
                dtype=DALIDataType.INT32,
            )
        )

    return tuple(outputs)
