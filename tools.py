#!/usr/bin/env python3

"""Overrides model and dataloader params to generate the full video"""
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import typer
from konductor.data import (
    DATASET_REGISTRY,
    ExperimentInitConfig,
    Split,
    get_dataloader,
    get_dataset_config,
)
from konductor.trainer.init import get_experiment_cfg
from konductor.utilities.pbar import LivePbar
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from torch import Tensor
from typing_extensions import Annotated

from src.dataset.common import MotionDatasetConfig
from src.model import MotionPerceiver
from src.model.motion_perceiver import MotionEncoder2Phase
from utils.eval_common import initialize, scenairo_id_tensor_2_str
from utils.visual import (
    reverse_image_transforms,
    write_flow_video,
    write_occupancy_video,
)

app = typer.Typer()


@dataclass
class EvalConfig:
    path: Path
    batch_size: int
    n_videos: int = 128
    video_thresh: float | None = None
    roi_scale: float = 1.0
    current_time_idx: int = 10
    sequence_length: int = 1
    time_stride: int = 1


def write_video_batch(
    data: Dict[str, Tensor],
    pred: Dict[str, Tensor],
    timestamps: List[float],
    path: Path,
    threshold: float | None = None,
    roadmap_scale: float = 1.0,
) -> None:
    """Write batch of videos"""
    mpool = mp.get_context("forkserver").Pool(processes=mp.cpu_count() * 3 // 4)
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
                    timestamps=timestamps,
                ),
            )

    if "flow" in pred:
        flow_path = path / "flow"
        flow_path.mkdir(parents=True, exist_ok=True)

        for b_idx, (p_flow, p_occ, t_flow) in enumerate(
            zip(pred["flow"], pred["heatmap"], data["flow"])
        ):
            # Super slow here for some reason when threadding? Maybe use of torch inside?
            mpool.apply_async(
                write_flow_video,
                kwds=dict(
                    pred_flow_sequence=p_flow,
                    pred_occ_sequence=p_occ.sigmoid(),
                    truth_flow_sequence=t_flow,
                    path=flow_path / f"{scenario_names[b_idx]}.webm",
                    mask_thresh=0.5,
                    timestamps=timestamps,
                ),
            )

    mpool.close()
    mpool.join()


def write_values(data: np.ndarray, path: Path) -> None:
    """Dump statistic to a text file for direct reading"""
    with open(path, "w", encoding="utf-8") as f:
        for elem in data:
            f.write(f"{elem}\n")


def reduce_gt_frames(
    frames: Tensor, phase_1_stride: int, transition: int, phase_2_stride: int
) -> Tensor:
    """Reduces the ground truth frames to match the predicted frames.

    :param frames: Ground truth occupancy frames of [b,c,t,h,w]
    :param phase_1_stride: Stride of the first phase
    :param transition: Index of transtion between phases
    :param phase_2_stride: Stride of the second phase
    """

    phase1 = frames[:, :, :transition:phase_1_stride]
    phase2 = frames[:, :, transition::phase_2_stride]
    return torch.cat([phase1, phase2], dim=2)


def generate_videos(
    model: MotionPerceiver, loader: DALIGenericIterator, config: EvalConfig
):
    """Create Videos of Inference"""
    with LivePbar(total=config.n_videos // config.batch_size) as pbar:
        for data in loader:
            n_samples = pbar.n * config.batch_size
            if n_samples >= config.n_videos:
                break

            data: Dict[str, Tensor] = data[0]  # remove list dimension
            outputs = model(**data)

            # Reformat the ground truth images
            if isinstance(model.encoder, MotionEncoder2Phase):
                reduce_keys = ["heatmap"]
                if "flow" in data:
                    reduce_keys.append("flow")

                for key in reduce_keys:
                    data[key] = reduce_gt_frames(
                        data[key],
                        model.encoder.stride_first,
                        model.encoder.transition_idx,
                        model.encoder.stride_second,
                    )
                    if key == "heatmap":
                        assert data[key].shape[2] == outputs[key].shape[1]
                    else:
                        assert data[key].shape[1] == outputs[key].shape[1]

                timestamps = list(
                    range(0, model.encoder.transition_idx, model.encoder.stride_first)
                )
                timestamps += list(
                    range(
                        model.encoder.transition_idx,
                        config.sequence_length,
                        model.encoder.stride_second,
                    )
                )
            else:
                timestamps = list(range(0, config.sequence_length, config.time_stride))

            # If the context is an rgb image, it is normalized so we
            # need to un-normalize for video writing and change to bgr
            if "roadmap" in data and data["roadmap"].shape[1] == 3:
                data["roadmap"] = reverse_image_transforms(data["roadmap"])

            write_video_batch(
                data,
                outputs,
                [(t - config.current_time_idx) / 10 for t in timestamps],
                config.path,
                config.video_thresh,
                config.roi_scale,
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


def override_dataset(exp_cfg: ExperimentInitConfig, new_dataset: str):
    """Override the original dataset while filtering unknown args"""
    new_dataset_type = DATASET_REGISTRY[new_dataset]
    new_kwargs = inspect.signature(new_dataset_type).parameters

    filtered_args: Dict[str, Any] = {}
    unused_args: Set[str] = set()
    for k, v in exp_cfg.data[0].dataset.args.items():
        if k in new_kwargs:
            filtered_args[k] = v
        else:
            unused_args.add(k)

    if len(unused_args) > 0:
        print(f"Unused kwargs in {new_dataset}: {unused_args}")

    exp_cfg.data[0].dataset.type = new_dataset
    exp_cfg.data[0].dataset.args = filtered_args


@app.command()
def make_video(
    run_path: Path,
    split: Annotated[Split, typer.Option()] = Split.VAL,
    n_samples: Annotated[int, typer.Option()] = 16,
    workers: Annotated[int, typer.Option()] = 4,
    batch_size: Annotated[int, typer.Option()] = 8,
    threshold: Annotated[Optional[float], typer.Option()] = None,
    dataset: Annotated[Optional[str], typer.Option()] = None,
) -> None:
    """"""
    exp_cfg = get_experiment_cfg(run_path.parent, None, run_path.name)
    exp_cfg.set_workers(workers)

    # Optional override dataset, filter out different kwargs
    if dataset is not None:
        override_dataset(exp_cfg, dataset)

    exp_cfg.set_batch_size(batch_size, split)

    model, data_cfg = initialize(exp_cfg)
    dataloader = get_dataloader(data_cfg, split)
    # model.encoder.input_indicies = set(range(0, 91, 10))

    eval_config = EvalConfig(
        exp_cfg.work_dir / exp_cfg.data[0].dataset.type / str(split.name.lower()),
        batch_size,
        n_samples,
        threshold,
        roi_scale=data_cfg.occupancy_roi,
        sequence_length=data_cfg.sequence_length,
        time_stride=data_cfg.time_stride,
        current_time_idx=data_cfg.current_time_idx,
    )

    with torch.inference_mode():
        generate_videos(model, dataloader, eval_config)


@app.command()
def visual_attention(
    run_path: Path,
    workers: Annotated[int, typer.Option()] = 4,
    n_samples: Annotated[int, typer.Option()] = 16,
    batch_size: Annotated[int, typer.Option()] = 8,
    threshold: Annotated[float, typer.Option()] = 0.0,
):
    exp_cfg = get_experiment_cfg(run_path.parent, None, run_path.name)
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


@app.command()
def get_params(run_path: Path):
    """Print number of parameters in model"""
    from konductor.models import get_model
    from torch import nn

    exp_cfg = get_experiment_cfg(run_path.parent, None, run_path.name)

    model: nn.Module = get_model(exp_cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"# Learnable Parameters: {total_params}")


if __name__ == "__main__":
    app()
