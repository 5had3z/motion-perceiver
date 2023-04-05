from dataclasses import dataclass, asdict, field
from subprocess import run
from multiprocessing import Pool, cpu_count
from pathlib import Path
import math
from typing import Any, Dict, List, Tuple

from nvidia.dali import pipeline_def, Pipeline
from nvidia.dali.types import DALIDataType, Constant
import nvidia.dali.math as dmath
import nvidia.dali.fn as fn
import nvidia.dali.tfrecord as tfrec

from konductor.modules.data import DATASET_REGISTRY, DatasetConfig, Mode


def _cache_record_idx(dataset_path: Path) -> Path:
    """
    Initially try to make with tf record dali index
    in folder adjacent to dataset suffixed by idx.
    If that fails due to permission requirements, make in /tmp.
    """
    dali_idx_path = dataset_path.parent / f"{dataset_path.name}_dali_idx"
    if not dali_idx_path.exists():
        try:
            dali_idx_path.mkdir()
            return dali_idx_path
        except OSError:
            print(
                f"Unable to create dali index at {dali_idx_path},"
                f" changing to /tmp/{dataset_path.name}_dali_idx"
            )

            dali_idx_path = Path(f"/tmp/{dataset_path.name}_dali_idx")
            if not dali_idx_path.exists():
                dali_idx_path.mkdir()

    return dali_idx_path


@dataclass
@DATASET_REGISTRY.register_module("waymo_motion")
class WaymoDatasetConfig(DatasetConfig):
    occupancy_size: int = field(kw_only=True)
    full_sequence: bool = False
    # fmt: off
    # Vehicle_features is order sensitive (this is ordering of channel concatenation)
    vehicle_features: List[str] = field(
        default_factory=lambda: [
            "x", "y", "bbox_yaw", "velocity_x", "velocity_y", "vel_yaw", "width", "length"])
    # fmt: on
    roadmap: bool = False
    signal_features: bool = False
    map_normalize: float = 0.0
    heatmap_time: List[int] | None = None
    filter_future: bool = False
    separate_classes: bool = False
    random_heatmap_minmax: Tuple[int, int] | None = None
    random_heatmap_count: int = 0
    scenario_id: bool = False
    use_sdc_frame: bool = False
    waymo_eval_frame: bool = False
    roadmap_size: int | None = None
    only_vehicles: bool = False
    # How to scale the occupancy roi, whole image => 1, center crop => 0.5
    occupancy_roi: float = 1.0

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
        if self.roadmap:
            output_map.append("roadmap")
        if self.signal_features:
            output_map.extend(["signals", "signals_valid"])
        if self.occupancy_size > 0:
            output_map.extend(["heatmap", "time_idx"])
        if self.scenario_id:
            output_map.append("scenario_id")

        return waymo_motion_pipe(root, cfg=self, **kwargs), output_map, root.stem


@pipeline_def
def waymo_motion_pipe(
    record_root: Path,
    shard_id: int,
    num_shards: int,
    random_shuffle: bool,
    cfg: WaymoDatasetConfig,
):
    """Waymo data should be split in separate folders
    training/validation/testing. Therefore we should be able
    to determine the split by the folder name"""

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

    tfrec_idx_root = _cache_record_idx(record_root)

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

    _time_keys = ["past", "current", "future"]

    if cfg.full_sequence:
        data = {
            "valid": fn.cat(
                *[
                    fn.cast(inputs[f"state/{time_}/valid"], dtype=DALIDataType.INT32)
                    for time_ in _time_keys
                ],
                axis=1,
            )
        }
        for key in cfg.vehicle_features:
            data[key] = fn.cat(
                *[inputs[f"state/{time_}/{key}"] for time_ in _time_keys], axis=1
            )
    else:
        data = {
            "valid": fn.cast(inputs["state/current/valid"], dtype=DALIDataType.INT32)
        }
        for key in cfg.vehicle_features:
            data[key] = inputs[f"state/current/{key}"]

    if cfg.use_sdc_frame:  # Center based on SDC
        sdc_mask = inputs["state/is_sdc"] * inputs["state/current/valid"][:, 0]
        sdc_mask = fn.cast(sdc_mask, dtype=DALIDataType.INT32)
        center_x = fn.masked_median(inputs["state/current/x"], sdc_mask)
        center_y = fn.masked_median(inputs["state/current/y"], sdc_mask)
        rot_angle = Constant(math.pi / 2, dtype=DALIDataType.FLOAT) - fn.masked_median(
            inputs["state/current/bbox_yaw"], sdc_mask
        )
    else:  #  Center system based on median agent position
        center_x = fn.masked_median(data["x"], data["valid"])
        center_y = fn.masked_median(data["y"], data["valid"])
        rot_angle = Constant(0.0, dtype=DALIDataType.FLOAT)

    # Center the frame
    data["x"] -= center_x
    data["y"] -= center_y

    if cfg.use_sdc_frame:
        # Rotate x,y coords
        temp_x = dmath.cos(rot_angle) * data["x"] - dmath.sin(rot_angle) * data["y"]
        data["y"] = dmath.sin(rot_angle) * data["x"] + dmath.cos(rot_angle) * data["y"]
        data["x"] = temp_x

        # Rotate vx,vy
        temp_x = (
            dmath.cos(rot_angle) * data["velocity_x"]
            - dmath.sin(rot_angle) * data["velocity_y"]
        )
        data["velocity_y"] = (
            dmath.sin(rot_angle) * data["velocity_x"]
            + dmath.cos(rot_angle) * data["velocity_y"]
        )
        data["velocity_x"] = temp_x

        data["bbox_yaw"] += rot_angle

    if cfg.waymo_eval_frame:  # flip y and move back 20m
        data["velocity_y"] *= -1
        data["y"] *= -1
        data["y"] += 20

    if cfg.map_normalize > 0.0:  # Divide by normalization factor
        for key in ["width", "length", "x", "y"]:
            data[key] /= cfg.map_normalize

        # x and y are within the ROI
        data["valid"] = (
            data["valid"] * (dmath.abs(data["x"]) < 1) * (dmath.abs(data["y"]) < 1)
        )

    if "bbox_yaw" in data:  # normalize angle bewteen -/+ pi
        data["bbox_yaw"] = dmath.atan2(
            dmath.sin(data["bbox_yaw"]), dmath.cos(data["bbox_yaw"])
        )

        if cfg.waymo_eval_frame:  # flip sign
            data["bbox_yaw"] *= -1

    # Maybe an idea for later, will stick with xy for now
    # if all(key in vehicle_features for key in ["velocity_x", "velocity_y"]):
    #     # kind of only an edge case when people don't traverse the way they're facing
    #     # and cars can only traverse the way they're facing unless they're cooked
    #     data["vel"] = dmath.sqrt(
    #         dmath.pow(data["velocity_x"], 2) + dmath.pow(data["velocity_y"], 2)
    #     )

    # Add type id to features at the end
    data["class"] = fn.stack(
        *[inputs["state/type"] for _ in range(91 if cfg.full_sequence else 1)], axis=1
    )
    if cfg.separate_classes:
        cfg.vehicle_features.append("class")

    if cfg.only_vehicles:  # Mark all other classes as invalid
        data["valid"] = data["valid"] * (data["class"] == 1)

    outputs = [
        fn.stack(*[data[k] for k in cfg.vehicle_features], axis=2),
        data["valid"],
    ]

    # roadmap into image
    if cfg.roadmap:
        outputs.append(
            fn.roadgraph_image(
                inputs[f"roadgraph_samples/xyz"],
                inputs[f"roadgraph_samples/type"],
                inputs[f"roadgraph_samples/id"],
                inputs[f"roadgraph_samples/valid"],
                center_x,
                center_y,
                rot_angle,
                size=cfg.roadmap_size,
                normalize_value=cfg.map_normalize,
                lane_center=True,
                waymo_eval_frame=cfg.waymo_eval_frame,
            )
        )

    # traffic signal into tensor [id, time]
    if cfg.signal_features:
        signal_valid = fn.cast(
            fn.cat(
                *[inputs[f"traffic_light_state/{time_}/valid"] for time_ in _time_keys],
                axis=0,
            ),
            dtype=DALIDataType.INT32,
        )
        signal_state = fn.cast(
            fn.cat(
                *[inputs[f"traffic_light_state/{time_}/state"] for time_ in _time_keys],
                axis=0,
            ),
            dtype=DALIDataType.FLOAT,
        )

        # Read signal positions over time
        signal_x = fn.cat(
            *[inputs[f"traffic_light_state/{time_}/x"] for time_ in _time_keys], axis=0
        )
        signal_y = fn.cat(
            *[inputs[f"traffic_light_state/{time_}/y"] for time_ in _time_keys], axis=0
        )

        # Center Signal Positions
        signal_x -= center_x
        signal_y -= center_y

        if cfg.use_sdc_frame:
            temp_x = dmath.cos(rot_angle) * signal_x - dmath.sin(rot_angle) * signal_y
            signal_y = dmath.sin(rot_angle) * signal_x + dmath.cos(rot_angle) * signal_y
            signal_x = temp_x

        if cfg.waymo_eval_frame:  # flip y and move back 20m
            signal_y *= -1
            signal_y += 20

        if cfg.map_normalize > 0.0:  # Normalize Signal Positions
            signal_x /= cfg.map_normalize
            signal_y /= cfg.map_normalize

            # Signals must be in map roi
            signal_valid = (
                signal_valid * (dmath.abs(signal_x) < 1) * (dmath.abs(signal_y) < 1)
            )

        outputs.extend(
            [fn.stack(signal_x, signal_y, signal_state, axis=2), signal_valid]
        )

    # Add occupancy heatmap
    if cfg.occupancy_size > 0:
        occ_kwargs = dict(
            size=cfg.occupancy_size,
            roi=cfg.occupancy_roi,
            const_time_idx=cfg.heatmap_time,
            filter_future=cfg.filter_future,
            separate_classes=cfg.separate_classes,
        )

        if cfg.random_heatmap_count > 0:
            assert (
                cfg.random_heatmap_minmax is not None
            ), "minmax must be specified if count > 0"
            occ_kwargs["n_random_idx"] = cfg.random_heatmap_count
            occ_kwargs["min_random_idx"] = cfg.random_heatmap_minmax[0]
            occ_kwargs["max_random_idx"] = cfg.random_heatmap_minmax[1]

        outputs.extend(
            fn.occupancy_mask(
                data["x"],
                data["y"],
                data["bbox_yaw"],
                data["width"],
                data["length"],
                data["valid"],
                data["class"],
                **occ_kwargs,
            )
        )

    if cfg.scenario_id:
        # Add padding since scenario_id contains <=16 characters
        pad_scenario_str = Constant(16 * [0], dtype=DALIDataType.UINT8)
        scenario_len = fn.shapes(inputs["scenario/id"])[0]
        scenario_pad = pad_scenario_str[:scenario_len] + inputs["scenario/id"]
        return tuple([o.gpu() for o in outputs] + [scenario_pad])

    return tuple([o.gpu() for o in outputs])
