#!/usr/bin/env python3

import logging
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from typing_extensions import Annotated

import torch
from konductor.metadata import Statistic
from konductor.trainer.init import init_data_manager, get_experiment_cfg
from konductor.trainer.pytorch import (
    AsyncFiniteMonitor,
    PyTorchTrainer,
    PyTorchTrainerConfig,
    PyTorchTrainerModules,
)
from konductor.utilities import comm
from konductor.utilities.pbar import PbarType, pbar_wrapper
import typer
from torch import Tensor
from torch.profiler import record_function

import src  # Imports all components into framework


class Trainer(PyTorchTrainer):
    def data_transform(self, data: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """Remove list dimension"""
        return data[0]

    def train_step(
        self, data: Dict[str, Tensor]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor] | None]:
        """
        Standard training step, if you don't want to calculate
        performance during training, return None for predictions.
        return
            Losses: description of losses for logging purposes
            Predictions: predictions in dict
        """
        with record_function("train_inference"):
            pred = self.modules.model(**data)

        with record_function("criterion"):
            losses = {}
            for criterion in self.modules.criterion:
                losses.update(criterion(pred, data))

        return losses, pred

    def val_step(
        self, data: Dict[str, Tensor]
    ) -> Tuple[Dict[str, Tensor] | None, Dict[str, Tensor]]:
        """
        Standard evaluation step, if you don't want to evaluate/track loss
        during evaluation, do not perform the calculation and return None
        in the loss part of the tuple.
        return:
            Losses: description of losses for logging purposes
            Predictions: predictions dict
        """
        with record_function("eval_inference"):
            pred = self.modules.model(**data)

        losses = {}
        for criterion in self.modules.criterion:
            losses.update(criterion(pred, data))

        return losses, pred


app = typer.Typer()


@app.command()
def main(
    workspace: Annotated[Path, typer.Option()],
    epoch: Annotated[int, typer.Option()],
    run_hash: Annotated[Optional[str], typer.Option()] = None,
    config_file: Annotated[Optional[Path], typer.Option()] = None,
    workers: Annotated[int, typer.Option()] = 4,
    pbar: Annotated[bool, typer.Option()] = False,
    brief: Annotated[Optional[str], typer.Option()] = None,
) -> None:
    """Main entrypoint to training model"""

    # Setup Training Configuration
    exp_config = get_experiment_cfg(workspace, config_file, run_hash)
    exp_config.set_workers(workers)

    # Initialize Training Modules
    train_modules = PyTorchTrainerModules.from_config(exp_config)

    # Setup Metadata Modules
    statistics: Dict[str, Statistic] = {
        "occupancy": src.statistics.Occupancy.from_config(exp_config)
    }
    if exp_config.model[0].args.get("signal_decoder", False):
        statistics["signal-forecast"] = src.statistics.Signal.from_config(exp_config)
    if exp_config.data[0].dataset.args.get("flow_mask", False):
        statistics["flow-predict"] = src.statistics.Flow.from_config(exp_config)
    data_manager = init_data_manager(exp_config, train_modules, statistics)
    if brief is not None:
        data_manager.metadata.brief = brief

    # Setup Trainer Configuration
    trainer_config = PyTorchTrainerConfig(**exp_config.trainer_kwargs)
    if pbar:
        trainer_config.pbar = partial(pbar_wrapper, pbar_type=PbarType.LIVE)
    else:
        trainer_config.pbar = partial(
            pbar_wrapper, pbar_type=PbarType.INTERVAL, fraction=0.1
        )

    trainer = Trainer(trainer_config, train_modules, data_manager)

    trainer.train(epoch=epoch)
    if isinstance(trainer.loss_monitor, AsyncFiniteMonitor):
        trainer.loss_monitor.stop()


if __name__ == "__main__":
    comm.initialize()
    torch.set_float32_matmul_precision("high")
    logging.basicConfig(
        format=f"%(asctime)s-RANK:{comm.get_local_rank()}-%(levelname)s-%(name)s: %(message)s",
        level=logging.INFO,
        force=True,
    )
    app()
