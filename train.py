from argparse import Namespace as NS
import logging
from typing import Tuple, Dict, List, Type

import torch
from torch import Tensor
from torch.profiler import record_function
from konductor.trainer.pbar import pbar_wrapper
from konductor.trainer.init import get_training_parser, init_training, cli_init_config
from konductor.trainer.pytorch import (
    PyTorchTrainer,
    PyTorchTrainerModules,
    PyTorchTrainerConfig,
    AsyncFiniteMonitor,
)
from konductor.metadata.statistics import Statistic

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

        return losses, None

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

        return None, pred


def setup(cli_args: NS) -> Trainer:
    exp_config = cli_init_config(cli_args)
    trainer_config = PyTorchTrainerConfig(loss_monitor=AsyncFiniteMonitor())
    if cli_args.pbar:
        trainer_config.pbar = pbar_wrapper

    statistics: Dict[str, Type[Statistic]] = {"occupancy": src.statistics.Occupancy}
    if exp_config.model[0].args.get("signal_decoder", False):
        statistics["signal-forecast"] = src.statistics.Signal

    return init_training(
        exp_config, Trainer, trainer_config, statistics, PyTorchTrainerModules
    )


def run(trainer: Trainer, epochs: int) -> None:
    """Run training until epochs is reached"""
    logging.info("Begin training")
    while trainer.data_manager.epoch < epochs:
        trainer.run_epoch()

    # Stop async thread
    if isinstance(trainer.loss_monitor, AsyncFiniteMonitor):
        trainer.loss_monitor.stop()


def main() -> None:
    cli_parser = get_training_parser()
    cli_parser.add_argument("--pbar", action="store_true")
    cli_args = cli_parser.parse_args()
    trainer = setup(cli_args)
    run(trainer, cli_args.epochs)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    logging.basicConfig(
        format="%(asctime)s-%(processName)s-%(levelname)s-%(name)s: %(message)s",
        level=logging.INFO,
    )
    main()
