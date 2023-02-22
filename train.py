from typing import Tuple, Dict, List

from torch import Tensor, nn
from torch.profiler import record_function
from konductor.trainer.initialisation import initialise_training
from konductor.trainer.pytorch import PyTorchTrainer

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

        return losses, pred

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


def main() -> None:
    trainer = initialise_training(Trainer, {"occupancy": Occupancy})
    trainer.run_epoch()


if __name__ == "__main__":
    main()
