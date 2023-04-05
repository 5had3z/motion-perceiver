"""Overrides model and dataloader params to generate the full video"""
import argparse
from functools import partial
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import multiprocessing as mp

import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from tqdm.auto import tqdm
from torch import nn, Tensor, inference_mode
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from src.dataset.waymo import WaymoDatasetConfig
from src.dataset.interaction import InteractionConfig
from src.statistics import Occupancy

from konductor.trainer.init import (
    parser_add_common_args,
    cli_init_config,
    get_model,
    get_dataloader,
    get_dataset_config,
)


@dataclass
class EvalConfig:
    path: Path
    batch_size: int
    n_statistic: int = 1024
    n_videos: int = 128
    video_thresh: Optional[float] = None
    filter_ids: Optional[Set[str]] = None


def scenairo_id_tensor_2_str(batch_ids: Tensor) -> List[str]:
    return ["".join([chr(c) for c in id_chars]) for id_chars in batch_ids]


def create_rgb_frame(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    resolution: Tuple[int, int],
    roadmap: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Create an rgb frame showing the ground truth and predicted frames"""
    bgr_frame = 255 * np.ones((*resolution, 3), dtype=np.uint8)
    prediction = cv2.resize(prediction, resolution, interpolation=cv2.INTER_LINEAR)
    ground_truth = cv2.resize(ground_truth, resolution, interpolation=cv2.INTER_NEAREST)

    # Red for false negatives
    bgr_frame[ground_truth == 1] = np.array((0, 0, 200), dtype=np.uint8)

    # Blue for false positives
    if threshold is not None:
        bgr_frame[prediction > threshold] = np.array((255, 0, 0), dtype=np.uint8)
    else:
        mask = ground_truth == 0
        rg = (255 * prediction).astype(np.uint8)[mask]
        b = np.zeros_like(rg)
        # subtract rg from prediction
        bgr_frame[mask] -= np.stack([b, rg, rg], axis=-1)

    # Green for true positives
    if threshold is not None:
        bgr_frame[(prediction > threshold) & (ground_truth == 1)] = np.array(
            (0, 255, 0), dtype=np.uint8
        )
    else:
        mask = (prediction > 0.5) & (ground_truth == 1)
        rb = ((1 - prediction) * 255).astype(np.uint8)[mask]
        g = 200 * np.ones_like(rb)
        bgr_frame[mask] = np.stack([rb, g, rb], axis=-1)

    if roadmap is not None:
        mask = roadmap > 0
        road_thresh = 255 * threshold if threshold is not None else 255 / 2
        for ch in range(3):
            mask &= bgr_frame[..., ch] > road_thresh  # mostly white
        bgr_frame[mask] = np.array((0, 0, 0), dtype=np.uint8)

    return bgr_frame


def write_video(
    data: np.ndarray,
    pred: np.ndarray,
    path: Path,
    roadmap: Optional[np.ndarray] = None,
    thresh: Optional[float] = None,
) -> None:
    """Write video of prediction over time"""

    video_shape = (800, 800)
    v_writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, video_shape
    )

    if not v_writer.isOpened():
        raise RuntimeError(f"Can't write video, writer not open: {path}")

    text_args = [cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA]

    if roadmap is not None:
        if len(roadmap.shape) == 3:
            roadmap = roadmap.squeeze(0)
        roadmap = cv2.resize(roadmap, video_shape, interpolation=cv2.INTER_NEAREST)

    for idx, (pred_frame, data_frame) in enumerate(zip(pred, data)):
        rgb_frame = create_rgb_frame(
            data_frame, pred_frame, video_shape, roadmap, thresh
        )
        if idx < 10:
            text = "past"
        elif idx == 10:
            text = "present"
        else:
            text = "future"
        text += f": {idx-10:+}"
        rgb_frame = cv2.putText(rgb_frame, text, (25, 50), *text_args)
        if thresh is not None:
            rgb_frame = cv2.putText(rgb_frame, f"pr>{thresh}", (25, 75), *text_args)

        v_writer.write(rgb_frame)

    v_writer.release()


def write_video_batch(
    data: Dict[str, Tensor],
    pred: Dict[str, Tensor],
    path: Path,
    global_it: int,
    threshold: Optional[float] = None,
) -> None:
    """Write batch of videos"""
    mpool = mp.Pool(processes=8)
    bz = data["heatmap"].shape[0]

    video_fn = (
        partial(write_video, thresh=threshold) if threshold is not None else write_video
    )

    if "roadmap" in data:
        roadmap_batch = data["roadmap"].cpu().numpy()
    else:
        roadmap_batch = [None for _ in range(bz)]

    for cls_idx, cls_name in enumerate(pred):
        for b_idx, (sample, pred_cls, roadmap) in enumerate(
            zip(data["heatmap"][:, cls_idx], pred[cls_name], roadmap_batch)
        ):
            mpool.apply_async(
                video_fn,
                kwds=dict(
                    data=sample.cpu().numpy(),
                    pred=pred_cls.sigmoid().cpu().numpy(),
                    roadmap=roadmap,
                    path=path / f"{cls_name}_occupancy_{global_it*bz + b_idx}.avi",
                ),
            )

    mpool.close()
    mpool.join()


def write_values(data: np.ndarray, path: Path) -> None:
    """Dump statistic to a text file for direct reading"""
    with open(path, "w", encoding="utf-8") as f:
        for elem in data:
            f.write(f"{elem}\n")


def plot_statistic_time(logger: Occupancy, path: Path, input_times: List[int]) -> None:
    assert logger.time_idxs is not None

    perf_data = logger.iteration_mean(0)

    for statistic in ["IoU", "AUC"]:
        plt.figure(statistic, figsize=(8, 4))
        plt.ylim((0, 1))
        plt.xlim((-1, len(logger.time_idxs)))
        plt.vlines(input_times, 0, 1, colors="green", label="observations", lw=5)
        plt.vlines(10, 0, 1, linestyles="dotted", colors="red", label="t=0", lw=5)

        data = np.empty(len(logger.time_idxs))
        for idx, time_idx in enumerate(logger.time_idxs):
            data[idx] = perf_data[f"{statistic}_{time_idx}"]
        plt.plot(data, lw=4)

        write_values(data, path / f"{statistic}.csv")

        plt.title(statistic)
        plt.xlabel("Time index (100ms inc)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path / f"{statistic}_over_sequence.png")


def gather_dict(meta_batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    ret_dict = {}
    for key in meta_batch[0]:
        if key == "scenario_id":
            ret_dict[key] = [m[key] for m in meta_batch]
        else:
            ret_dict[key] = torch.cat([m[key] for m in meta_batch], dim=0)
    return ret_dict


def yield_filtered_batch(dataloader, filter_ids: Set[str], batch_size: int):
    meta_batch = []
    for batch in dataloader:
        batch = batch[0]
        assert batch["scenario_id"].shape[0] == 1, "Batch size greater than 1"
        id_str = scenairo_id_tensor_2_str(batch["scenario_id"])[0]
        if id_str not in filter_ids:
            continue
        batch["scenario_id"] = id_str

        meta_batch.append(batch)
        if len(meta_batch) == batch_size:
            yield [gather_dict(meta_batch)]
            meta_batch = []

    if len(meta_batch) != 0:
        yield [gather_dict(meta_batch)]


@inference_mode()
def statistic_evaluation(
    model: nn.Module,
    loader: DALIGenericIterator,
    logger: Occupancy,
    config: EvalConfig,
) -> None:
    """"""
    vid_path = config.path / "occupancy_video"

    if not vid_path.exists():
        vid_path.mkdir(parents=True)

    max_samples = max(config.n_videos, config.n_statistic)
    # if not set run over whole dataset for statistics
    # if we're using the filter_ids, run over that dataset
    if max_samples == 0 or config.filter_ids is not None:
        max_samples = (
            len(loader) * config.batch_size
            if config.filter_ids is None
            else len(config.filter_ids)
        )
        config.n_statistic = max_samples

    # save now as it'll be lost after jit
    input_indicies = list(model.encoder.input_indicies)

    # fmt: off
    arg_order = [
        "time_idx", "agents", "agents_valid",
        "roadgraph", "roadgraph_valid", "roadmap",
        "signals", "signals_valid",
    ]
    # fmt: on

    if config.filter_ids is not None:
        loader = yield_filtered_batch(loader, config.filter_ids, config.batch_size)

    with tqdm(total=max_samples // config.batch_size) as pbar:
        for data in loader:
            n_samples = pbar.n * config.batch_size
            if n_samples >= max_samples:
                break

            data: Dict[str, Tensor] = data[0]  # remove list dimension

            pos_args = []
            for arg in arg_order:
                pos_args.append(data[arg] if arg in data else torch.empty([]))
            pos_args = tuple(pos_args)

            # if not isinstance(model, torch.jit.ScriptModule):
            #     print("Tracing Pytorch Model to improve speed")
            #     model = torch.jit.trace(model, pos_args, strict=False)

            outputs = model(*pos_args)
            for key in outputs:
                outputs[key][outputs[key] < 0] *= 8.0

            if n_samples < config.n_videos:
                write_video_batch(data, outputs, vid_path, pbar.n, config.video_thresh)

            if n_samples < config.n_statistic:
                logger(0, outputs, data)

            pbar.update(1)

    if config.n_statistic > 0:
        logger.flush()
        plot_statistic_time(logger, config.path, input_indicies)


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add additonal evaluation settings to parser"""
    parser.add_argument("--n_videos", type=int, default=128)
    parser.add_argument("--n_statistic", type=int, default=1024)
    parser.add_argument("--video_thresh", type=float)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--dataset_override", type=str, help="override trained dataset for testing"
    )
    parser.add_argument(
        "--scenario_id",
        type=Path,
        help="Optional path to sequence ids to run evaluation on",
    )


def initialize() -> Tuple[nn.Module, DALIGenericIterator, Occupancy, EvalConfig]:
    """Initalise eval for motion perciever"""
    parser = argparse.ArgumentParser()
    parser_add_common_args(parser)
    add_eval_args(parser)
    parser.add_argument("-w", "--workers", type=int, default=4)
    args = parser.parse_args()

    exp_cfg = cli_init_config(args)
    exp_cfg.model[0].optimizer.args.pop("step_interval", None)

    if args.dataset_override:
        print(f"Overriding {exp_cfg.data[0].dataset.type} to {args.dataset_override}")
        exp_cfg.data[0].dataset.type = args.dataset_override

    dataset_cfg: WaymoDatasetConfig | InteractionConfig = get_dataset_config(exp_cfg)
    model: nn.Module = get_model(exp_cfg)
    ckpt = torch.load(
        exp_cfg.work_dir / "latest.pt",
        map_location=f"cuda:{torch.cuda.current_device()}",
    )["model"]

    try:
        model.load_state_dict(ckpt)
    except RuntimeError:
        # probably weights from old framework
        for k in list(ckpt.keys()):
            ckpt[k.removeprefix("model.")] = ckpt.pop(k)
        ckpt.pop("decoder.output")
        model.load_state_dict(ckpt)

    model = model.eval().cuda()

    if isinstance(dataset_cfg, WaymoDatasetConfig):
        dataset_cfg.heatmap_time = list(range(0, 91))
        dataset_cfg.filter_future = True
        dataset_cfg.waymo_eval_frame = True
    elif isinstance(dataset_cfg, InteractionConfig):
        dataset_cfg.heatmap_time = list(range(0, 40))
    else:
        raise NotImplementedError(f"Unknown dataset {dataset_cfg}")

    dataset_cfg.random_heatmap_count = 0

    if args.scenario_id:
        assert isinstance(dataset_cfg, WaymoDatasetConfig), "Only waymo has scenario id"
        exp_cfg.data[0].val_loader.args["batch_size"] = 1
        # exp_config.dataset.args.heatmap_time = list(range(20, 91, 10))
        dataset_cfg.scenario_id = True
        # Load challenge ids
        with open(args.scenario_id, "r", encoding="utf-8") as f:
            filter_ids = set([l.strip() for l in f.readlines()])
    else:
        exp_cfg.data[0].val_loader.args["batch_size"] = args.batch_size
        filter_ids = None

    dataloader: DALIGenericIterator = get_dataloader(dataset_cfg, "val")

    thresh: Optional[float] = args.video_thresh if args.video_thresh else None
    eval_config = EvalConfig(
        exp_cfg.work_dir / exp_cfg.data[0].dataset.type,
        args.batch_size,
        args.n_statistic,
        args.n_videos,
        thresh,
        filter_ids,
    )

    logger = Occupancy.from_config(
        1000, eval_config.path / "occupancy.parquet", **dataset_cfg.properties
    )

    return model, dataloader, logger, eval_config


def main() -> None:
    """"""
    args = initialize()
    statistic_evaluation(*args)


if __name__ == "__main__":
    main()
