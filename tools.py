#!/usr/bin/env python3

"""Overrides model and dataloader params to generate the full video"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing_extensions import Annotated
from typing import Dict, Tuple, Optional
import multiprocessing as mp

import cv2
import numpy as np
import typer
import torch
from torch import Tensor
from torchvision.transforms.functional import normalize
from torchvision.utils import flow_to_image
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from konductor.utilities.pbar import LivePbar
from konductor.trainer.init import get_experiment_cfg, get_dataset_config

from src.dataset.common import MotionDatasetConfig
from src.model import MotionPerceiver
from utils.eval_common import initialize, scenairo_id_tensor_2_str
from utils.visual import write_occupancy_video, apply_ts_text

app = typer.Typer()


@dataclass
class EvalConfig:
    path: Path
    batch_size: int
    n_videos: int = 128
    video_thresh: float | None = None
    roi_scale: float = 1.0
    time_stride: int = 1


def create_flow_frame(
    pred_flow: np.ndarray,
    pred_occ: np.ndarray,
    truth_flow: np.ndarray,
    frame_size: Tuple[int, int],
    mask_thresh=0.5,
) -> np.ndarray:
    """Create a side-by-side frame of predicted occupancy flow and ground truth,
    mask out predicted flow with predicted occupancy over a threshold
    A threshold of zero is obviously no threshold (show all flow for every pixel)
    """
    pred_flow_rgb = flow_to_image(torch.tensor(pred_flow)).numpy()
    pred_flow_rgb[:, pred_occ < mask_thresh] = 255  # set to white
    truth_flow_rgb = flow_to_image(torch.tensor(truth_flow)).numpy()
    rgb_frame = cv2.hconcat(
        [np.moveaxis(pred_flow_rgb, 0, 2), np.moveaxis(truth_flow_rgb, 0, 2)]
    )
    rgb_frame = cv2.resize(rgb_frame, frame_size, interpolation=cv2.INTER_LINEAR)
    return rgb_frame


def write_flow_video(
    pred_flow_sequence: np.ndarray,
    pred_occ_sequence: np.ndarray,
    truth_flow_sequence: np.ndarray,
    path: Path,
    mask_thresh=0.5,
):
    """"""
    video_shape = (1600, 800)
    v_writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"VP90"), 10, video_shape
    )

    if not v_writer.isOpened():
        raise RuntimeError(f"Can't write video, writer not open: {path}")

    # Check if prediction is 2phase, if so squeeze ground truth
    if pred_flow_sequence.shape[1] == 19:
        truth_flow_sequence[:, 11:19] = truth_flow_sequence[:, 20::10]
        truth_flow_sequence = truth_flow_sequence[:, :19]

    for idx in range(pred_flow_sequence.shape[1]):
        rgb_frame = create_flow_frame(
            pred_flow_sequence[:, idx],
            pred_occ_sequence[idx],
            truth_flow_sequence[:, idx],
            video_shape,
            mask_thresh,
        )

        rgb_frame = apply_ts_text(idx, rgb_frame, extra=f"pr>{mask_thresh}")

        v_writer.write(rgb_frame)

    v_writer.release()


def write_video_batch(
    data: Dict[str, Tensor],
    pred: Dict[str, Tensor],
    path: Path,
    threshold: float | None = None,
    roadmap_scale: float = 1.0,
    time_stride: int = 1,
) -> None:
    """Write batch of videos"""
    mpool = mp.Pool(processes=mp.cpu_count() // 2)
    bz = data["heatmap"].shape[0]

    occ_path = path / "occupancy"
    occ_path.mkdir(parents=True, exist_ok=True)

    roadmap_batch = data["roadmap"].cpu().numpy() if "roadmap" in data else [None] * bz
    signals_batch = (
        [data["signals"], data["signals_valid"].bool()] if "signals" in data else None
    )
    scenario_names = scenairo_id_tensor_2_str(data["scenario_id"])

    for cls_idx, cls_name in enumerate(p for p in pred if "heatmap" in p):
        for b_idx, (sample, pred_cls, roadmap) in enumerate(
            zip(data["heatmap"][:, cls_idx], pred[cls_name], roadmap_batch)
        ):
            if signals_batch is not None:
                signals = [
                    x[b_idx].cpu().transpose(1, 0).numpy() for x in signals_batch
                ]
            else:
                signals = None

            mpool.apply_async(
                write_occupancy_video,
                kwds=dict(
                    data=sample.cpu().numpy(),
                    pred=pred_cls.sigmoid().cpu().numpy(),
                    signals=signals,
                    roadmap=roadmap,
                    path=occ_path / f"{scenario_names[b_idx]}_{cls_name}.webm",
                    roadmap_scale=roadmap_scale,
                    thresh=threshold,
                    time_stride=time_stride,
                ),
            )

    if "flow" in pred:
        flow_path = path / "flow"
        flow_path.mkdir(parents=True, exist_ok=True)

        for b_idx, (p_flow, p_occ, t_flow) in enumerate(
            zip(pred["flow"], pred["heatmap"], data["flow"])
        ):
            # Super slow here for some reason when threadding? Maybe use of torch inside?
            # mpool.apply_async(
            write_flow_video(
                # kwds=dict(
                pred_flow_sequence=p_flow.cpu().numpy(),
                pred_occ_sequence=p_occ.sigmoid().cpu().numpy(),
                truth_flow_sequence=t_flow.cpu().numpy(),
                path=flow_path / f"{scenario_names[b_idx]}.webm",
                mask_thresh=0.5,
                # ),
            )

    mpool.close()
    mpool.join()


def write_values(data: np.ndarray, path: Path) -> None:
    """Dump statistic to a text file for direct reading"""
    with open(path, "w", encoding="utf-8") as f:
        for elem in data:
            f.write(f"{elem}\n")


def generate_videos(
    model: MotionPerceiver,
    loader: DALIGenericIterator,
    config: EvalConfig,
) -> None:
    """"""
    with LivePbar(total=config.n_videos // config.batch_size) as pbar:
        for data in loader:
            n_samples = pbar.n * config.batch_size
            if n_samples >= config.n_videos:
                break

            data: Dict[str, Tensor] = data[0]  # remove list dimension
            outputs = model(**data)
            for key in outputs:
                outputs[key][outputs[key] < 0] *= 8.0

            # If the context is an rgb image, it is normalized so we
            # need to un-normalize for video writing and change to bgr
            if "roadmap" in data and data["roadmap"].shape[1] == 3:
                normalize(
                    data["roadmap"],
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                    inplace=True,
                )
                data["roadmap"] = (
                    (255 * data["roadmap"][:, [2, 1, 0]]).clamp(0, 255).to(torch.uint8)
                )

            write_video_batch(
                data,
                outputs,
                config.path,
                config.video_thresh,
                config.roi_scale,
                config.time_stride,
            )

            pbar.update(1)


def visualise_output_attention(
    model: MotionPerceiver, loader: DALIGenericIterator, config: EvalConfig
):
    """Visualise the attention for the output pixels of the model
    to show that each token has an individual response, we can use
    this to track objects over time and show that each latent variable
    represents a single agent"""

    def attn_hook(module, inputs, outputs: Tuple[Tensor, Tensor]) -> None:
        """Hook that grabs the attention map and writes to disk"""
        _, attn = outputs

        for bidx, data in enumerate(attn):
            bfolder = config.path / f"sample_{bidx}"
            bfolder.mkdir(exist_ok=True)
            for tkn_idx in range(data.shape[-1]):
                (bfolder / f"token_{tkn_idx}").mkdir(exist_ok=True)

            for tkn_idx in range(data.shape[-1]):
                img = (
                    (255.0 * data[..., tkn_idx].view(256, 256))
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
                img = cv2.resize(img, (768, 768), interpolation=cv2.INTER_LINEAR)
                tfolder = bfolder / f"token_{tkn_idx}"
                t_idx = len(list(tfolder.iterdir()))
                cv2.imwrite(str(tfolder / f"{t_idx:02}.png"), img)

        # np.savez_compressed(
        #     config.path / "attn",
        #     **{str(i): v.cpu().numpy() for i, v in enumerate(attn)},
        # )

    model.decoder.cross_attention._modules[
        "0"
    ].attention.attention.register_forward_hook(attn_hook)

    for data in loader:
        data = data[0]
        model(**data)
        break


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add additonal evaluation settings to parser"""
    parser.add_argument("--n_videos", type=int, default=128)
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


@app.command()
def make_video(
    path: Path,
    n_samples: Annotated[int, typer.Option()] = 16,
    workers: Annotated[int, typer.Option()] = 4,
    batch_size: Annotated[int, typer.Option()] = 8,
    threshold: Annotated[Optional[float], typer.Option()] = None,
    dataset: Annotated[Optional[str], typer.Option()] = None,
) -> None:
    """"""
    exp_cfg = get_experiment_cfg(path.parent, None, path.name)
    exp_cfg.set_workers(workers)

    # Optional override dataset
    if dataset is not None:
        exp_cfg.data[0].dataset.type = dataset
    exp_cfg.set_batch_size(batch_size, "val")

    data_cfg: MotionDatasetConfig = get_dataset_config(exp_cfg)
    model, dataloader = initialize(exp_cfg, "val")

    eval_config = EvalConfig(
        exp_cfg.work_dir / exp_cfg.data[0].dataset.type,
        batch_size,
        n_samples,
        threshold,
        data_cfg.occupancy_roi,
        data_cfg.time_stride,
    )

    with torch.inference_mode():
        generate_videos(model, dataloader, eval_config)


@app.command()
def visual_attention(
    path: Path,
    workers: Annotated[int, typer.Option()] = 4,
    n_samples: Annotated[int, typer.Option()] = 16,
    batch_size: Annotated[int, typer.Option()] = 8,
    threshold: Annotated[float, typer.Option()] = 0.0,
):
    exp_cfg = get_experiment_cfg(path.parent, None, path.name)
    exp_cfg.set_workers(workers)
    exp_cfg.set_batch_size(batch_size, "val")

    data_cfg: MotionDatasetConfig = get_dataset_config(exp_cfg)

    model, dataloader = initialize(exp_cfg, "val")

    eval_config = EvalConfig(
        exp_cfg.work_dir / exp_cfg.data[0].dataset.type,
        batch_size,
        n_samples,
        threshold,
        data_cfg.occupancy_roi,
        data_cfg.time_stride,
    )
    with torch.inference_mode():
        visualise_output_attention(model, dataloader, eval_config)


if __name__ == "__main__":
    app()
