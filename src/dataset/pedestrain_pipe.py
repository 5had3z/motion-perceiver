from pathlib import Path
from typing import List

import numpy as np
import nvidia.dali.math as dmath
import nvidia.dali.tfrecord as tfrec
from konductor.init import ModuleInitConfig
from nvidia.dali import fn, newaxis, pipeline_def
from nvidia.dali.types import Constant, DALIDataType, DALIInterpType

try:
    from .common import (
        MotionDatasetConfig,
        dali_rad2deg,
        get_sample_idxs,
        get_tfrecord_cache,
    )
except ImportError:
    from common import (
        MotionDatasetConfig,
        dali_rad2deg,
        get_sample_idxs,
        get_tfrecord_cache,
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

    tfrecord_caches = get_tfrecord_cache(record_root, tfrecords)

    inputs = fn.readers.tfrecord(
        path=[str(record_root / r) for r in tfrecords],
        index_path=tfrecord_caches,
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

    if cfg.roadmap:
        # Since the coordinate origin is the the top-left of
        # the context image, we must change that to the center
        ctx_image, source_px_per_m = fn.load_scene(
            inputs["scenario_id"],
            src=str(cfg.basepath / "images"),
            metadata=str(cfg.basepath / "images" / "image_metadata.yml"),
        )
        ctx_image = fn.decoders.image(ctx_image, device="cpu")
        img_shape = (
            fn.shapes(ctx_image, dtype=DALIDataType.FLOAT)[:-1] / 2 / -source_px_per_m
        )
        center = fn.stack(img_shape[1], img_shape[0])
    elif any(a.type == "center" for a in augmentations):
        center = -fn.cat(
            fn.masked_median(inputs["x"], data_valid),
            fn.masked_median(inputs["y"], data_valid),
        )
    else:
        center = Constant([0.0, 0.0], dtype=DALIDataType.FLOAT)

    xy_tf = fn.transforms.combine(fn.transforms.translation(offset=center), rot_mat)

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
    if all(k in cfg.vehicle_features for k in ["width", "length"]):
        assert hasattr(cfg, "fake_size"), "fake_size needs to be set"
        data_out.append(fn.full_like(data_xy, cfg.fake_size))
    if "class" in cfg.vehicle_features:
        # Create class vector
        data_class = fn.cast(inputs["type"], dtype=DALIDataType.FLOAT)
        data_class = fn.stack(
            *[data_class] * cfg.sequence_length if cfg.full_sequence else 1, axis=1
        )
        data_class = data_class[:, :, newaxis]
        data_out.append(data_class)

    outputs = [fn.cat(*data_out, axis=2), inputs["valid"]]

    if cfg.time_stride > 1:
        outputs = fn.stride_slice(outputs, axis=1, stride=cfg.time_stride)

    if cfg.roadmap:
        # Normalize Pixel Per Meter
        target_px_per_m = cfg.roadmap_size / (cfg.map_normalize * 2)
        rescale_ratio = target_px_per_m / source_px_per_m
        img_shape = fn.shapes(ctx_image, dtype=DALIDataType.FLOAT)[:-1]
        scaled_shape = img_shape * rescale_ratio
        ctx_image = fn.resize(ctx_image, size=scaled_shape)

        # Center the image
        img_shape = fn.shapes(ctx_image, dtype=DALIDataType.FLOAT)[:-1]
        out_shape = Constant(
            (cfg.roadmap_size, cfg.roadmap_size), dtype=DALIDataType.FLOAT
        )
        centering = fn.stack(*[img_shape[i] - cfg.roadmap_size for i in [1, 0]])
        ctx_image = fn.warp_affine(
            ctx_image,
            fn.transforms.translation(offset=centering / 2),
            size=out_shape,
            interp_type=DALIInterpType.INTERP_LINEAR,
            inverse_map=True,
        )

        # Transform affine from meters to pixels
        xy_rot = fn.transforms.rotation(
            angle=fn.reshape(dali_rad2deg(angle_rad), shape=[]), center=out_shape / 2
        )

        # Apply Warp Affine
        ctx_image = fn.warp_affine(
            ctx_image,
            xy_rot,
            interp_type=DALIInterpType.INTERP_LINEAR,
            inverse_map=False,
        )

        # Transpose image to CHW and Normalize
        ctx_image = fn.cast(
            fn.transpose(ctx_image, perm=[2, 0, 1]), dtype=DALIDataType.FLOAT
        ) / Constant(255, dtype=DALIDataType.FLOAT)
        ctx_image = fn.normalize(
            ctx_image,
            mean=Constant(
                [0.485, 0.456, 0.406], dtype=DALIDataType.FLOAT, shape=[3, 1, 1]
            ),
            stddev=Constant(
                [0.229, 0.224, 0.225], dtype=DALIDataType.FLOAT, shape=[3, 1, 1]
            ),
        )

        outputs.append(ctx_image)

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
