# Run the model with inserted "updates" of the sdc's future position
from dataclasses import dataclass
from pathlib import Path
from typing_extensions import Annotated

import typer
import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from konductor.data import get_dataloader, get_dataset_config
from konductor.trainer.init import get_experiment_cfg, add_workers_to_dataloader

from src.model import MotionPerceiver
from src.dataset.waymo import WaymoDatasetConfig
from utils.eval_common import load_model

app = typer.Typer()


@dataclass
class EvalConfig:
    random: bool  # Sample car randomly or always sdc


def run(model: MotionPerceiver, loader: DALIGenericIterator, config: EvalConfig):
    """"""
    for data in loader:
        sample = data[0]
        pred = model(**sample)


@app.command()
def main(
    path: Path,
    batch_size: Annotated[int, typer.Option()] = 4,
    random: Annotated[bool, typer.Option()] = False,
):
    """Perform inference where either the SDC or a
    random car's timesteps are added as future
    measurements so that we are essentially conditioning
    our prediction on the action taken by some agent"""
    exp_cfg = get_experiment_cfg(path.parent, None, path.name)
    add_workers_to_dataloader(exp_cfg, 4)

    dataset: WaymoDatasetConfig = get_dataset_config(exp_cfg)
    dataset.sdc_index = not random
    dataset.random_heatmap_count = 0
    dataset.random_heatmap_piecewise.clear()
    dataset.heatmap_time = list(range(0, 90 // dataset.time_stride + 1))
    dataset.val_loader.args["batch_size"] = batch_size

    dataloader = get_dataloader(dataset, "val")

    model = load_model(exp_cfg)
    config = EvalConfig(random)
    with torch.inference_mode():
        run(model, dataloader, config)


if __name__ == "__main__":
    app()
