# Run the model with inserted "updates" of the sdc's future position
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing_extensions import Annotated

import typer
import torch
from torch import Tensor
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from konductor.data import get_dataloader, get_dataset_config
from konductor.trainer.init import get_experiment_cfg
from konductor.utilities.pbar import LivePbar

from src.model import MotionPerceiver
from src.dataset.waymo import WaymoDatasetConfig
from utils.eval_common import load_model, scenairo_id_tensor_2_str
from utils.visual import write_occupancy_video

app = typer.Typer()


@dataclass
class EvalConfig:
    path: Path  # Path to experiment
    random: bool  # Sample car randomly or always sdc
    n_samples: int  # Number of samples to run from dataloader


def mask_future_data_except_target(data: Dict[str, Tensor]):
    """Mask the future vehicle and signal data except a single
    vehicle so that every future state only includes a vehicle
    measurement that we want to condition on"""
    data["agents_valid"][..., 11:] = (
        data["sdc_mask"].unsqueeze(-1).expand_as(data["agents_valid"][..., 11:])
    )
    data["signals_valid"][..., 11:] = torch.zeros_like(data["signals_valid"][..., 11:])


def run(model: MotionPerceiver, loader: DALIGenericIterator, config: EvalConfig):
    """"""
    input_stride = model.encoder.input_indicies[0] - model.encoder.input_indicies[0]
    model.encoder.input_indicies = range(0, 91, input_stride)
    write_dir = config.path / "conditional"
    write_dir.mkdir(exist_ok=True)

    with LivePbar(total=config.n_samples, desc="Cond. Infer") as pbar:
        for data in loader:
            if pbar.n > config.n_samples:
                return
            sample: Dict[str, Tensor] = data[0]
            mask_future_data_except_target(sample)
            pred: Tensor = model(**sample)["heatmap"][0]
            pred[pred < 0] *= 8.0

            signals = [
                x[0].cpu().transpose(1, 0).numpy()
                for x in [sample["signals"], sample["signals_valid"].bool()]
            ]

            filename = scenairo_id_tensor_2_str(sample["scenario_id"])[0] + ".webm"
            write_occupancy_video(
                data=sample["heatmap"][0, 0].cpu().numpy(),
                pred=pred.sigmoid().cpu().numpy(),
                path=write_dir / filename,
                signals=signals,
                roadmap=sample["roadmap"][0].cpu().numpy(),
                roadmap_scale=0.5,
            )
            pbar.update(1)


@app.command()
def main(
    path: Path,
    batch_size: Annotated[int, typer.Option()] = 1,
    random: Annotated[bool, typer.Option()] = False,
    n_samples: Annotated[int, typer.Option()] = 16,
):
    """Perform inference where either the SDC or a
    random car's timesteps are added as future
    measurements so that we are essentially conditioning
    our prediction on the action taken by some agent"""
    exp_cfg = get_experiment_cfg(path.parent, None, path.name)
    exp_cfg.set_workers(4)
    exp_cfg.set_batch_size(batch_size, "val")

    dataset: WaymoDatasetConfig = get_dataset_config(exp_cfg)
    dataset.sdc_index = not random
    dataset.random_heatmap_count = 0
    dataset.random_heatmap_piecewise.clear()
    dataset.heatmap_time = list(range(0, 90 // dataset.time_stride + 1))
    dataset.scenario_id = True

    dataloader = get_dataloader(dataset, "val")

    model = load_model(exp_cfg).eval()
    config = EvalConfig(exp_cfg.work_dir, random, n_samples)
    with torch.inference_mode():
        run(model, dataloader, config)


if __name__ == "__main__":
    app()
