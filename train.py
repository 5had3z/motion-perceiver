from konductor.trainer.initialisation import initialise_training
from konductor.trainer.pytorch import PyTorchTrainer
import src


def main() -> None:
    trainer = initialise_training(PyTorchTrainer)
    trainer.run_epoch()


if __name__ == "__main__":
    main()
