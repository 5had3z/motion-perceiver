"""Test to check for parity with native waymo-open-motion data"""
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from konductor.data.dali import DaliLoaderConfig
from waymo_open_dataset.utils import occupancy_flow_data, occupancy_flow_grids

from src.dataset.waymo import WaymoDatasetConfig, waymo_motion_pipe
from utils.export_tf import get_waymo_task_config
from utils.eval_common import scenairo_id_tensor_2_str

loadpath = Path(__file__).parent / "waymo-sample.tfrecord"
assert loadpath.exists()


@pytest.fixture
def tf_dataloader():
    dataset = (
        tf.data.TFRecordDataset(str(loadpath.resolve()))
        .map(occupancy_flow_data.parse_tf_example)
        .batch(4)
    )
    return dataset


def get_tf_waypoints(data: dict[str, tf.Tensor]):
    task_cfg = get_waymo_task_config()
    data = occupancy_flow_data.add_sdc_fields(data)
    timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(
        data, task_cfg
    )
    waypoints_grids = occupancy_flow_grids.create_ground_truth_waypoint_grids(
        timestep_grids, task_cfg
    )
    return waypoints_grids


@pytest.fixture
def dali_dataloader():
    cfg = WaymoDatasetConfig(
        train_loader=DaliLoaderConfig(4),
        val_loader=DaliLoaderConfig(4),
        occupancy_size=256,
        heatmap_time=list(range(20, 91, 10)),
        occupancy_roi=0.5,
        map_normalize=80.0,
        flow_mask=True,
        flow_type="history",
        scenario_id=True,
        full_sequence=True,
        use_sdc_frame=True,
        waymo_eval_frame=True,
        only_vehicles=True,
    )

    pipeline = waymo_motion_pipe(
        loadpath.parent, cfg=cfg, **cfg.train_loader.pipe_kwargs()
    )
    loader = cfg.train_loader.get_instance(
        pipeline,
        ["agents", "agents_valid", "time_idx", "heatmamp", "flow", "scenario_id"],
        reader_name=loadpath.parent.stem,
    )
    return loader


def test_loading_tf(tf_dataloader):
    for data in tf_dataloader:
        waypoints = get_tf_waypoints(data)


def test_loading_dali(dali_dataloader):
    for sample in dali_dataloader:
        sample = sample[0]


def test_similarity(tf_dataloader, dali_dataloader):
    tf_data = next(iter(tf_dataloader))
    tf_waypoints = get_tf_waypoints(tf_data)
    tf_scenarios = list(tf_data["scenario/id"].numpy().astype(str))
    dali_data = next(iter(dali_dataloader))[0]
    dali_scenarios = scenairo_id_tensor_2_str(dali_data["scenario_id"])
    assert tf_scenarios == dali_scenarios

    tf_flow = np.stack([f.numpy() for f in tf_waypoints.vehicles.flow])  # [T,B,H,W,C]
    tf_flow = np.moveaxis(tf_flow, [1, 4], [0, 1])
    dali_flow = dali_data["flow"].cpu().numpy()  # [B,C,T,H,W]
    assert tf_flow.shape == dali_flow.shape
    diff = np.abs(tf_flow - dali_flow)

    # from matplotlib import pyplot as plt
    # import os

    # for t_idx in range(8):
    #     plt.subplot(231)
    #     plt.imshow(tf_flow[0, 0, t_idx])
    #     plt.subplot(232)
    #     plt.imshow(dali_flow[0, 0, t_idx])
    #     plt.subplot(233)
    #     plt.imshow(diff[0, 0, t_idx])
    #     plt.subplot(234)
    #     plt.imshow(tf_flow[0, 1, t_idx])
    #     plt.subplot(235)
    #     plt.imshow(dali_flow[0, 1, t_idx])
    #     plt.subplot(236)
    #     plt.imshow(diff[0, 1, t_idx])
    #     plt.suptitle(
    #         f"{t_idx=}, max diff: x:{diff[0, 0, t_idx].max()}, y:{diff[0, 1, t_idx].max()}"
    #     )
    #     plt.savefig(f"{os.environ['IMAGE_OUT']}/waymo_diff_{t_idx}.png")
