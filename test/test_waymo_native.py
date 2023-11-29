"""Test to check for parity with native waymo-open-motion data"""
from pathlib import Path

import pytest
import tensorflow as tf
from konductor.data.dali import DaliLoaderConfig
from waymo_open_dataset.utils import occupancy_flow_data, occupancy_flow_grids

from src.dataset.waymo import WaymoDatasetConfig, waymo_motion_pipe
from utils.export_tf import get_waymo_task_config

loadpath = Path(__file__).parent / "waymo-sample.tfrecord"
assert loadpath.exists()


@pytest.fixture
def tf_dataloader():
    dataset = (
        tf.data.TFRecordDataset(str(loadpath.resolve()))
        .map(occupancy_flow_data.parse_tf_example)
        .batch(4)
    )
    task_cfg = get_waymo_task_config()
    return dataset, task_cfg


@pytest.fixture
def dali_dataloader():
    cfg = WaymoDatasetConfig(
        train_loader=DaliLoaderConfig(1, workers=4),
        val_loader=DaliLoaderConfig(1),
        occupancy_size=256,
        heatmap_time=list(range(10, 91, 10)),
        occupancy_roi=0.5,
        map_normalize=80.0,
        flow_mask=True,
        flow_type="history",
        scenario_id=True,
        full_sequence=True,
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
    dataset, task_cfg = tf_dataloader
    for data in dataset:
        data = occupancy_flow_data.add_sdc_fields(data)
        timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(
            data, task_cfg
        )
        waypoints_grids = occupancy_flow_grids.create_ground_truth_waypoint_grids(
            timestep_grids, task_cfg
        )


def test_loading_dali(dali_dataloader):
    for sample in dali_dataloader:
        print(sample[0])
