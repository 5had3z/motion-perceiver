from pathlib import Path
import subprocess
from copy import deepcopy
from typing import Callable, Dict, List
import os
import zlib
from math import ceil

from tqdm.auto import tqdm
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from google.protobuf import text_format
from waymo_open_dataset.utils.occupancy_flow_grids import (
    WaypointGrids,
    _WaypointGridsOneType,
)
from waymo_open_dataset.protos import occupancy_flow_submission_pb2
from waymo_open_dataset.utils import (
    occupancy_flow_data,
    occupancy_flow_grids,
    occupancy_flow_metrics,
)
from waymo_open_dataset.protos.occupancy_flow_metrics_pb2 import (
    OccupancyFlowTaskConfig,
    OccupancyFlowMetrics,
)

from .eval_data import MetricData


def get_waymo_task_config():
    config = OccupancyFlowTaskConfig()
    config_text = """
    num_past_steps: 10
    num_future_steps: 80
    num_waypoints: 8
    cumulative_waypoints: false
    normalize_sdc_yaw: true
    grid_height_cells: 256
    grid_width_cells: 256
    sdc_y_in_grid: 192
    sdc_x_in_grid: 128
    pixels_per_meter: 3.2
    agent_points_per_side_length: 48
    agent_points_per_side_width: 16
    """
    text_format.Parse(config_text, config)
    return config


def _make_submission_proto() -> occupancy_flow_submission_pb2.ChallengeSubmission:
    """Makes a submission proto to store predictions for one shard."""
    submission = occupancy_flow_submission_pb2.ChallengeSubmission()
    submission.account_name = "bryce.ferenczi@monash.edu"
    submission.unique_method_name = "MotionPerceiver"
    submission.authors.extend(["Bryce Ferenczi", "Michael Burke", "Tom Drummond"])
    submission.description = (
        "Latent state representation of the scene is learned. The future latent state "
        "of the scene is predicted based on the current latent state. Latent state can "
        "be contiuously updated with new information from the scene resulting in a "
        "realtime streaming architecture."
    )
    submission.method_link = "https://github.com/5had3z/motion-perceiver"
    return submission


def _get_pred_waypoint_logits(
    predictions: tf.Tensor,
) -> occupancy_flow_grids.WaypointGrids:
    """Slices model predictions into occupancy and flow grids to [b,h,w,c]."""
    pred_waypoint_logits = occupancy_flow_grids.WaypointGrids()

    # prediction should be [b,t,h,w,c]
    b, t, h, w, c = predictions.shape
    flow_buffer = tf.zeros((b, h, w, 2))
    ocl_occ_buffer = tf.zeros((b, h, w, 1))

    # Slice channels into output predictions.
    for tidx in range(t):
        prediction = predictions[:, tidx]
        pred_waypoint_logits.vehicles.observed_occupancy.append(
            prediction[..., 0, tf.newaxis]
        )
        if c == 3:
            flow_buffer = prediction[..., 1:]
        pred_waypoint_logits.vehicles.flow.append(flow_buffer)
        pred_waypoint_logits.vehicles.occluded_occupancy.append(ocl_occ_buffer)

    return pred_waypoint_logits


def _add_waypoints_to_scenario_prediction(
    pred_waypoints: occupancy_flow_grids.WaypointGrids,
    scenario_prediction: occupancy_flow_submission_pb2.ScenarioPrediction,
    config: OccupancyFlowTaskConfig,
) -> None:
    """Add predictions for all waypoints to scenario_prediction message."""
    for k in range(config.num_waypoints):
        waypoint_message = scenario_prediction.waypoints.add()
        # Observed occupancy.
        obs_occupancy = pred_waypoints.vehicles.observed_occupancy[k].numpy()
        obs_occupancy_quantized = np.round(obs_occupancy * 255).astype(np.uint8)
        obs_occupancy_bytes = zlib.compress(obs_occupancy_quantized.tobytes())
        waypoint_message.observed_vehicles_occupancy = obs_occupancy_bytes
        # Occluded occupancy.
        occ_occupancy = pred_waypoints.vehicles.occluded_occupancy[k].numpy()
        occ_occupancy_quantized = np.round(occ_occupancy * 255).astype(np.uint8)
        occ_occupancy_bytes = zlib.compress(occ_occupancy_quantized.tobytes())
        waypoint_message.occluded_vehicles_occupancy = occ_occupancy_bytes
        # Flow.
        flow = pred_waypoints.vehicles.flow[k].numpy()
        flow_quantized = np.clip(np.round(flow), -128, 127).astype(np.int8)
        flow_bytes = zlib.compress(flow_quantized.tobytes())
        waypoint_message.all_vehicles_flow = flow_bytes


def _apply_sigmoid_to_occupancy_logits(
    pred_waypoint_logits: WaypointGrids,
) -> WaypointGrids:
    """Converts occupancy logits to probabilities."""
    pred_waypoints = WaypointGrids()
    pred_waypoints.vehicles.observed_occupancy = [
        tf.sigmoid(x) for x in pred_waypoint_logits.vehicles.observed_occupancy
    ]
    pred_waypoints.vehicles.occluded_occupancy = [
        tf.sigmoid(x) for x in pred_waypoint_logits.vehicles.occluded_occupancy
    ]
    pred_waypoints.vehicles.flow = pred_waypoint_logits.vehicles.flow
    return pred_waypoints


def _load_prediction(numpy_file: Path) -> np.ndarray:
    np_pred: np.ndarray = np.load(numpy_file)
    # Post process step on heatmap logits
    np_pred[0, np_pred[0] < 0] *= 2.0
    if np_pred.shape[0] > 1:  # Scale flow from m/s to pix/frame
        np_pred[1:] *= -3.2
    np_pred = np.moveaxis(np_pred, 0, -1)
    return np_pred


def _read_prediction_file(numpy_file: Path) -> WaypointGrids:
    """Read numpy file with logits and apply any extra transforms"""
    np_pred = _load_prediction(numpy_file)[None]  # Unsqueeze batch dim
    prediction = tf.convert_to_tensor(np_pred)
    pred_waypoint_logits = _get_pred_waypoint_logits(prediction)
    return pred_waypoint_logits


def _add_predictions_to_submission(
    config: OccupancyFlowTaskConfig,
    submission: occupancy_flow_submission_pb2.ChallengeSubmission,
    inference_blob_folder: Path,
) -> None:
    """Iterate over all test examples in one shard and generate predictions."""
    numpy_files = list(inference_blob_folder.iterdir())

    with tqdm(total=len(numpy_files)) as pbar:
        for numpy_file in numpy_files:
            # Run inference.
            pred_waypoint_logits = _read_prediction_file(numpy_file)
            pred_waypoints = _apply_sigmoid_to_occupancy_logits(pred_waypoint_logits)

            # Make new scenario prediction message.
            scenario_prediction = submission.scenario_predictions.add()
            scenario_prediction.scenario_id = numpy_file.stem

            # Add all waypoints.
            _add_waypoints_to_scenario_prediction(
                pred_waypoints, scenario_prediction, config
            )
            pbar.update(1)


def export_evaluation(pred_path: Path):
    dev = tf.config.list_physical_devices("GPU")
    if len(dev) > 0:
        tf.config.experimental.set_memory_growth(dev[0], True)

    task_config = get_waymo_task_config()
    submission = _make_submission_proto()
    _add_predictions_to_submission(task_config, submission, pred_path)
    binary_path = pred_path.parent / "occupancy_flow_submission.binproto"

    print(f"Saving {len(submission.scenario_predictions)} predictions to {binary_path}")
    with open(binary_path, "wb") as f:
        f.write(submission.SerializeToString())

    tar_path = pred_path.parent / "submission.tar.gz"
    subprocess.run(["tar", "czvf", str(tar_path), str(binary_path)])


def write_images(
    true_waypoints: List[tf.Tensor],
    pred_waypoints: List[tf.Tensor],
    scenario_id: str,
    metrics: OccupancyFlowMetrics,
    opt_pred_waypoints: List[tf.Tensor] | None = None,
    opt_metrics: OccupancyFlowMetrics | None = None,
) -> None:
    # Write image of prediction vs loaded ground truth
    subplot_base = 121 if opt_pred_waypoints is None else 131

    for t_idx, (truth, pred) in enumerate(zip(true_waypoints, pred_waypoints)):
        plt.figure(figsize=(12 if opt_pred_waypoints is None else 18, 7))
        plt.subplot(subplot_base)
        plt.imshow(truth[0, :, :, 0])
        plt.title("waymo ground truth")
        plt.subplot(subplot_base + 1)
        plt.title("pytorch prediction")
        plt.imshow(pred[0, :, :, 0].numpy())
        suptitle_str = f"+{t_idx+1}sec TF IoU: {metrics.vehicles_observed_iou:.3f}"
        suptitle_str += f", AUC: {metrics.vehicles_observed_auc:.3f}"

        if opt_pred_waypoints is not None:
            assert opt_metrics is not None
            plt.subplot(subplot_base + 2)
            plt.title("pytorch ground truth")
            plt.imshow(opt_pred_waypoints[t_idx][0, :, :, 0].numpy())
            suptitle_str += f"    Torch IoU: {opt_metrics.vehicles_observed_iou:.3f}"
            suptitle_str += f", AUC: {opt_metrics.vehicles_observed_auc:.3f}"

        plt.suptitle(suptitle_str)
        plt.tight_layout()
        plt.savefig(f"{scenario_id}_{t_idx}.png")
        plt.close()


def _sample_waypoint_one_type(src: _WaypointGridsOneType, idx: int):
    dest = _WaypointGridsOneType()
    dest.observed_occupancy = [src.observed_occupancy[idx]]
    dest.occluded_occupancy = [src.occluded_occupancy[idx]]
    dest.flow = [src.flow[idx]]
    if len(src.flow_origin_occupancy) > 0:
        dest.flow_origin_occupancy = [src.flow_origin_occupancy[idx]]
    return dest


def sample_waypoint(waypoints: WaypointGrids, idx: int) -> WaypointGrids:
    """Make WaypointGrids with single waypoint at idx"""
    waypoint = WaypointGrids()
    waypoint.vehicles = _sample_waypoint_one_type(waypoints.vehicles, idx)

    # Only copy cyclists and pedestrians if they exist
    if len(waypoints.cyclists.observed_occupancy) > 0:
        waypoint.cyclists = _sample_waypoint_one_type(waypoints.cyclists, idx)
    if len(waypoints.pedestrians.observed_occupancy) > 0:
        waypoint.pedestrians = _sample_waypoint_one_type(waypoints.pedestrians, idx)

    return waypoint


def _evaluate_timepoints_and_mean(
    dataset: tf.data.Dataset,
    config: OccupancyFlowTaskConfig,
    inference_blob_folder: Path,
    split: str,
    pbar: tqdm | None = None,
) -> List[MetricData]:
    """Evaluate model and performance at waypoints as well as the average"""
    perf: List[MetricData] = [MetricData("waypoint_mean")]
    for waypoint in range(config.num_waypoints):
        perf.append(MetricData(f"waypoint_{waypoint + 1}"))

    cfg_keyframe = deepcopy(config)
    cfg_keyframe.num_waypoints = 1

    for inputs in dataset:
        inputs = occupancy_flow_data.add_sdc_fields(inputs)
        timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(
            inputs, config
        )
        true_waypoints = occupancy_flow_grids.create_ground_truth_waypoint_grids(
            timestep_grids, config
        )

        pred_waypoint_logits = _get_pred_waypoint_logits(inputs["prediction"])
        pred_waypoints = _apply_sigmoid_to_occupancy_logits(pred_waypoint_logits)

        metrics = occupancy_flow_metrics.compute_occupancy_flow_metrics(
            config, true_waypoints, pred_waypoints
        )

        perf[0].add_auc(metrics.vehicles_observed_auc)
        perf[0].add_iou(metrics.vehicles_observed_iou)
        perf[0].add_epe(metrics.vehicles_flow_epe)

        for waypoint in range(config.num_waypoints):
            true_waypoint = sample_waypoint(true_waypoints, waypoint)
            pred_waypoint = sample_waypoint(pred_waypoints, waypoint)
            metrics = occupancy_flow_metrics.compute_occupancy_flow_metrics(
                cfg_keyframe, true_waypoint, pred_waypoint
            )
            perf[waypoint + 1].add_auc(metrics.vehicles_observed_auc)
            perf[waypoint + 1].add_iou(metrics.vehicles_observed_iou)
            perf[waypoint + 1].add_epe(metrics.vehicles_flow_epe)

        if pbar is not None:
            pbar.update(1)
            if pbar.n % 10 == 0:
                pbar.set_description(str(perf[0]))

    return perf


def _get_validation_and_prediction(
    dataset: tf.data.Dataset,
    config: OccupancyFlowTaskConfig,
    inference_blob_folder: Path,
    split: str,
    visualize: bool = False,
    pbar: tqdm | None = None,
) -> List[MetricData]:
    """Iterate over all test examples in one shard and generate predictions."""
    pt_stats = MetricData("pytorch")
    tf_stats = MetricData("tensorflow")

    for inputs in dataset:
        inputs = occupancy_flow_data.add_sdc_fields(inputs)
        timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(
            inputs, config
        )
        true_waypoints = occupancy_flow_grids.create_ground_truth_waypoint_grids(
            timestep_grids, config
        )

        pred_waypoint_logits = _get_pred_waypoint_logits(inputs["prediction"])
        pred_waypoints = _apply_sigmoid_to_occupancy_logits(pred_waypoint_logits)

        tf_metrics = occupancy_flow_metrics.compute_occupancy_flow_metrics(
            config, true_waypoints, pred_waypoints
        )

        tf_stats.add_auc(tf_metrics.vehicles_observed_auc)
        tf_stats.add_iou(tf_metrics.vehicles_observed_iou)
        tf_stats.add_epe(tf_metrics.vehicles_flow_epe)

        scenario_id = inputs["scenario/id"].numpy().astype(str)[0]
        true_waypoints_ = deepcopy(true_waypoints)
        pytorch_gt_file = (
            inference_blob_folder.parent.parent
            / f"{split}_ground_truth"
            / f"{scenario_id}.npy"
        )
        # only care about observed occupancy
        true_waypoints_.vehicles.observed_occupancy = _read_prediction_file(
            pytorch_gt_file
        ).vehicles.observed_occupancy

        torch_metrics = occupancy_flow_metrics.compute_occupancy_flow_metrics(
            config, true_waypoints_, pred_waypoints
        )

        pt_stats.add_auc(torch_metrics.vehicles_observed_auc)
        pt_stats.add_iou(torch_metrics.vehicles_observed_iou)
        pt_stats.add_epe(-1.0)

        if visualize:
            write_images(
                true_waypoints.vehicles.observed_occupancy,
                pred_waypoints.vehicles.observed_occupancy,
                scenario_id,
                tf_metrics,
                true_waypoints_.vehicles.observed_occupancy,
                torch_metrics,
            )

        if pbar is not None:
            pbar.update(1)
            if pbar.n % 10 == 0:
                desc = f"{pt_stats} - {tf_stats}"
                pbar.set_description(desc)

    return [pt_stats, tf_stats]


def evaluate_methods(
    id_path: Path, pred_path: Path, split, eval_fn: Callable
) -> List[MetricData]:
    dev = tf.config.list_physical_devices("GPU")
    if len(dev) > 0:
        tf.config.experimental.set_memory_growth(dev[0], True)

    split_ = {"test": "testing", "val": "validation"}[split.name]

    task_config = get_waymo_task_config()
    data_shards = (
        Path(os.environ.get("DATAPATH", "/data"))
        / "waymo-motion"
        / "tf_example"
        / split_
        / f"{split_}_tfexample.tfrecord*"
    )

    filenames = tf.io.matching_files(str(data_shards))

    with tf.io.gfile.GFile(id_path) as f:
        scenario_ids = tf.constant([id.rstrip() for id in f.readlines()])

    def valid_scenario_id(data: Dict[str, tf.Tensor]):
        """TF doesn't have hashmap lookup which is lame"""
        return tf.reduce_any(tf.equal(data["scenario/id"], scenario_ids))

    def read_file(data: Dict[str, tf.Tensor]):
        """Load numpy prediction as tensor"""

        def load_numpy_data(filename):
            filename = filename.numpy().decode("utf-8")
            return _load_prediction(pred_path / f"{filename}.npy")

        data["prediction"] = tf.py_function(
            load_numpy_data, [data["scenario/id"]], tf.float32
        )

        return data

    batch_size = 1
    dataset = (
        tf.data.TFRecordDataset(filenames)
        .map(occupancy_flow_data.parse_tf_example, num_parallel_calls=4)
        .filter(valid_scenario_id)
        .map(read_file, num_parallel_calls=4)
        .prefetch(4)
        .batch(batch_size)
    )

    with tqdm(total=ceil(len(scenario_ids) / batch_size)) as pbar:
        stats = eval_fn(dataset, task_config, pred_path, split, pbar=pbar)

    return stats
