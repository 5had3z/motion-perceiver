from pathlib import Path
import subprocess
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Sequence, Optional
import os
import zlib

from tqdm.auto import tqdm
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from google.protobuf import text_format
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
        "Scene is encoded into a latent space which can be updated and evolved"
    )
    submission.method_link = ""
    return submission


def _get_pred_waypoint_logits(
    predictions: tf.Tensor,
) -> occupancy_flow_grids.WaypointGrids:
    """Slices model predictions into occupancy and flow grids to [b,h,w,c]."""
    pred_waypoint_logits = occupancy_flow_grids.WaypointGrids()

    # prediction should be [t,c,h,w]
    flow_buffer = tf.zeros((1, *predictions.shape[-2:], 2))
    ocl_occ_buffer = tf.zeros((1, *predictions.shape[-2:], 1))

    # Slice channels into output predictions.
    for waypoint_prediction in predictions:
        pred_waypoint_logits.vehicles.observed_occupancy.append(
            waypoint_prediction[None, ..., None]
        )
        pred_waypoint_logits.vehicles.occluded_occupancy.append(ocl_occ_buffer)
        pred_waypoint_logits.vehicles.flow.append(flow_buffer)

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
    pred_waypoint_logits: occupancy_flow_grids.WaypointGrids,
) -> occupancy_flow_grids.WaypointGrids:
    """Converts occupancy logits to probabilities."""
    pred_waypoints = occupancy_flow_grids.WaypointGrids()
    pred_waypoints.vehicles.observed_occupancy = [
        tf.sigmoid(x) for x in pred_waypoint_logits.vehicles.observed_occupancy
    ]
    pred_waypoints.vehicles.occluded_occupancy = [
        tf.sigmoid(x) for x in pred_waypoint_logits.vehicles.occluded_occupancy
    ]
    pred_waypoints.vehicles.flow = pred_waypoint_logits.vehicles.flow
    return pred_waypoints


def _read_prediction_file(numpy_file: Path) -> occupancy_flow_grids.WaypointGrids:
    """Read numpy file with logits and apply any extra transforms"""
    prediction = np.load(numpy_file)
    # Post process step on logits
    prediction[prediction < 0] *= 8.0
    prediction = tf.convert_to_tensor(prediction)

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
    task_config = get_waymo_task_config()
    submission = _make_submission_proto()
    _add_predictions_to_submission(task_config, submission, pred_path)
    binary_path = pred_path.parent / "occupancy_flow_submission.binproto"

    print(f"Saving {len(submission.scenario_predictions)} predictions to {binary_path}")
    with open(binary_path, "wb") as f:
        f.write(submission.SerializeToString())

    tar_path = pred_path.parent / "submission.tar.gz"
    subprocess.run(["tar", "czvf", str(tar_path), str(binary_path)])


@dataclass
class EvalStatistics:
    name: str
    _iou: List[float] = field(default_factory=list)
    _auc: List[float] = field(default_factory=list)

    def add_iou(self, iou: float) -> None:
        self._iou.append(iou)

    def add_auc(self, auc: float) -> None:
        self._auc.append(auc)

    @property
    def auc(self) -> float:
        return np.array(self._auc).mean()

    @property
    def iou(self) -> float:
        return np.array(self._iou).mean()

    def __str__(self) -> str:
        return f"{self.name} - IoU: {self.iou:.3f}, AUC: {self.auc:.3f}"


def write_images(
    true_waypoints: List[tf.Tensor],
    pred_waypoints: List[tf.Tensor],
    scenario_id: str,
    metrics: OccupancyFlowMetrics,
    opt_pred_waypoints: Optional[List[tf.Tensor]] = None,
    opt_metrics: Optional[OccupancyFlowMetrics] = None,
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


def _get_validation_and_prediction(
    test_dataset: tf.data.Dataset,
    test_scenario_ids: Sequence[str],
    inference_blob_folder: Path,
    config: OccupancyFlowTaskConfig,
    split: str,
    visualize: bool = False,
) -> None:
    """Iterate over all test examples in one shard and generate predictions."""
    pt_stats = EvalStatistics("pytorch")
    tf_stats = EvalStatistics("tensorflow")

    with tqdm(total=len(test_scenario_ids)) as pbar:
        for inputs in test_dataset:
            scenario_id = inputs["scenario/id"].numpy().astype(str)[0]
            if scenario_id not in test_scenario_ids:
                continue
            pytorch_gt_file = (
                inference_blob_folder.parent.parent
                / f"{split}_ground_truth"
                / f"{scenario_id}.npy"
            )

            inputs = occupancy_flow_data.add_sdc_fields(inputs)
            timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(
                inputs, config
            )
            true_waypoints = occupancy_flow_grids.create_ground_truth_waypoint_grids(
                timestep_grids, config
            )

            # Load Prediction.
            pred_waypoint_logits = _read_prediction_file(
                inference_blob_folder / f"{scenario_id}.npy"
            )
            pred_waypoints = _apply_sigmoid_to_occupancy_logits(pred_waypoint_logits)

            tf_metrics = occupancy_flow_metrics.compute_occupancy_flow_metrics(
                config, true_waypoints, pred_waypoints
            )

            tf_stats.add_auc(tf_metrics.vehicles_observed_auc)
            tf_stats.add_iou(tf_metrics.vehicles_observed_iou)

            true_waypoints_ = deepcopy(true_waypoints)
            # only care about observed occupancy
            true_waypoints_.vehicles.observed_occupancy = _read_prediction_file(
                pytorch_gt_file
            ).vehicles.observed_occupancy

            true_waypoints_.vehicles.observed_occupancy = [
                true_waypoints_.vehicles.observed_occupancy[0][0, 0, idx, tf.newaxis]
                for idx in range(
                    true_waypoints_.vehicles.observed_occupancy[0].shape[2]
                )
            ]
            torch_metrics = occupancy_flow_metrics.compute_occupancy_flow_metrics(
                config, true_waypoints_, pred_waypoints
            )

            pt_stats.add_auc(torch_metrics.vehicles_observed_auc)
            pt_stats.add_iou(torch_metrics.vehicles_observed_iou)

            if visualize:
                write_images(
                    true_waypoints.vehicles.observed_occupancy,
                    pred_waypoints.vehicles.observed_occupancy,
                    scenario_id,
                    tf_metrics,
                    true_waypoints_.vehicles.observed_occupancy,
                    torch_metrics,
                )

            pbar.update(1)
            if pbar.n % 10 == 0:
                pbar.set_description(f"{pt_stats} - {tf_stats}")

    print(f"{pt_stats}\n{tf_stats}")


def evaluate_methods(
    id_path: Path, pred_path: Path, split: str, visualize: bool = False
):
    task_config = get_waymo_task_config()
    data_shards = (
        Path(os.environ.get("DATAPATH", "/data"))
        / split
        / f"{split}_tfexample.tfrecord*"
    )

    filenames = tf.io.matching_files(str(data_shards))
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(occupancy_flow_data.parse_tf_example)
    dataset = dataset.batch(1)

    with tf.io.gfile.GFile(id_path) as f:
        test_scenario_ids = f.readlines()
        test_scenario_ids = [id.rstrip() for id in test_scenario_ids]

    _get_validation_and_prediction(
        dataset, test_scenario_ids, pred_path, task_config, split, visualize
    )
