#!/usr/bin/env python3

import logging
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import typer
import yaml
from konductor.init import ExperimentInitConfig, ModuleInitConfig
from konductor.metadata import DataManager, Statistic
from konductor.trainer.pytorch import (
    AsyncFiniteMonitor,
    PyTorchTrainer,
    PyTorchTrainerConfig,
    PyTorchTrainerModules,
)
from konductor.utilities import comm
from konductor.utilities.pbar import PbarType, pbar_wrapper
from konductor.utilities.profiler import make_default_profiler_kwargs, profile_function
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.profiler import record_function, schedule
from typing_extensions import Annotated

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


class ImageNetTrainer(PyTorchTrainer):
    """Handle image-label pair a bit differently"""

    def data_transform(self, data: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """Remove list dimension"""
        return data[0]

    def train_step(self, data) -> Tuple[Dict[str, Tensor] | None, ...]:
        pred = self.modules.model(data["image"])
        losses = {}
        for criterion in self.modules.criterion:
            losses.update(criterion(pred, data["label"][:, 0]))
        return losses, {"pred": pred}

    def val_step(self, data) -> Tuple[Dict[str, Tensor] | None, ...]:
        pred = self.modules.model(data["image"])
        losses = {}
        for criterion in self.modules.criterion:
            losses.update(criterion(pred, data["label"][:, 0]))

        if isinstance(self.modules.scheduler, ReduceLROnPlateau):
            self.plateau_loss.update(sum(losses.values()).item())

        return losses, {"pred": pred}


def get_statistics(exp_config: ExperimentInitConfig):
    """Get statistics used for training based on dataset and algorithm configurations"""
    statistics: Dict[str, Statistic] = {}
    if exp_config.data[0].dataset.type == "imagenet-1k":
        statistics["classification"] = src.statistics.Classification.from_config(
            exp_config
        )
    else:
        statistics["occupancy"] = src.statistics.Occupancy.from_config(exp_config)
        if "signal_decoder" in exp_config.model[0].args:
            statistics["signal-forecast"] = src.statistics.Signal.from_config(
                exp_config
            )
        if "flow_mask" in exp_config.data[0].dataset.args:
            statistics["flow-predict"] = src.statistics.Flow.from_config(exp_config)

    return statistics


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
    remote: Annotated[Optional[Path], typer.Option()] = None,
    profile: Annotated[bool, typer.Option()] = False,
) -> None:
    """Main entrypoint to training model"""

    if run_hash is not None:
        assert config_file is None, "run-hash and config-file are exclusive"
        exp_config = ExperimentInitConfig.from_run(workspace / run_hash)
    elif config_file is not None:
        exp_config = ExperimentInitConfig.from_config(workspace, config_file)
    else:
        raise RuntimeError("run-hash or config-file must be specified")
    exp_config.set_workers(workers)

    if remote is not None:
        with open(remote, "r", encoding="utf-8") as file:
            remote_cfg = yaml.safe_load(file)
        exp_config.remote_sync = ModuleInitConfig(**remote_cfg)

    # Initialize Training Modules
    train_modules = PyTorchTrainerModules.from_config(exp_config)

    # Setup Metadata Modules
    data_manager = DataManager.default_build(
        exp_config, train_modules.get_checkpointables(), get_statistics(exp_config)
    )
    if brief is not None:
        data_manager.metadata.brief = brief

    # Setup Trainer Configuration
    trainer_config = PyTorchTrainerConfig(**exp_config.trainer)
    if pbar and comm.get_local_rank() == 0:
        trainer_config.pbar = partial(pbar_wrapper, pbar_type=PbarType.LIVE)
    elif comm.get_local_rank() == 0:
        trainer_config.pbar = partial(
            pbar_wrapper, pbar_type=PbarType.INTERVAL, fraction=0.1
        )

    if exp_config.data[0].dataset.type == "imagenet-1k":
        trainer = ImageNetTrainer(trainer_config, train_modules, data_manager)
    else:
        trainer = Trainer(trainer_config, train_modules, data_manager)

    if profile:
        prof_kwargs = make_default_profiler_kwargs(save_dir=data_manager.workspace)
        prof_kwargs["schedule"] = schedule(wait=1, warmup=2, active=5, repeat=1)
        profile_function(trainer._train, profile_kwargs=prof_kwargs)
    else:
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
