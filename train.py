from typing import Tuple, Dict, List

from torch import Tensor, nn, no_grad
from torch.profiler import record_function
from konductor.trainer.pbar import pbar_wrapper
from konductor.trainer.initialisation import get_training_parser, initialise_training
from konductor.trainer.pytorch import (
    PyTorchTrainer,
    PytorchTrainingModules,
    PerfLogger,
    TrainingMangerConfig,
)

import src  # Imports all components into framework
from src.statistics import Occupancy


class Trainer(PyTorchTrainer):
    @staticmethod
    def train_step(
        batch_data: List[Dict[str, Tensor]],
        model: nn.Module,
        criterions: List[nn.Module],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor] | None]:
        """
        Standard training step, if you don't want to calculate
        performance during training, return None for predictions.
        return
            Losses: description of losses for logging purposes
            Predictions: predictions in dict
        """
        data = {k: d.cuda() for k, d in batch_data[0].items()}

        with record_function("train_inference"):
            pred = model(**data)

        with record_function("criterion"):
            losses = {}
            for criterion in criterions:
                losses.update(criterion(pred, data))

        return losses, None

    @staticmethod
    def val_step(
        batch_data: List[Dict[str, Tensor]],
        model: nn.Module,
        criterions: List[nn.Module],
    ) -> Tuple[Dict[str, Tensor] | None, Dict[str, Tensor]]:
        """
        Standard evaluation step, if you don't want to evaluate/track loss
        during evaluation, do not perform the calculation and return None
        in the loss part of the tuple.
        return:
            Losses: description of losses for logging purposes
            Predictions: predictions dict
        """
        data = {k: d.cuda() for k, d in batch_data[0].items()}

        with record_function("eval_inference"):
            pred = model(**data)

        return None, pred

    @staticmethod
    @no_grad()
    @record_function("statistics")
    def log_step(
        logger: PerfLogger,
        data: List[Dict[str, Tensor]],
        preds: Dict[str, Tensor] | None,
        losses: Dict[str, Tensor] | None,
    ) -> None:
        """
        Logging things, statistics should have "losses" tracker, all losses are forwarded
        to that. If losses are missing logging of them will be skipped (if you don't want
        to log loss during eval). If predictions are missing then accuracy logging will
        be skipped (if you don't want to log acc during training)
        """
        for statistic in logger.logger_keys:
            if statistic == "loss" and losses is not None:
                logger.log(statistic, {k: v.item() for k, v in losses.items()})
            elif preds is not None:
                logger.log(statistic, preds, data[0])


def main() -> None:
    cli_parser = get_training_parser()
    cli_parser.add_argument("--pbar", action="store_true")
    cli_args = cli_parser.parse_args()
    trainer_config = TrainingMangerConfig()
    if cli_args.pbar:
        trainer_config.pbar = pbar_wrapper
    trainer = initialise_training(
        cli_args,
        Trainer,
        trainer_config,
        {"occupancy": Occupancy},
        PytorchTrainingModules,
    )
    trainer.run_epoch()


if __name__ == "__main__":
    main()
