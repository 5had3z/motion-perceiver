"""Export pytorch predictions to numpy file
to then import to waymo"""

from argparse import Namespace
import enum
from pathlib import Path
import os
from contextlib import closing


import typer
from konductor.utilities.metadata import Metadata

from utils.eval_data import update_eval_db, get_db


class Mode(str, enum.Enum):
    train = "train"
    val = "val"
    test = "test"


app = typer.Typer()


def get_id_path(split: str):
    ext_split = {"test": "testing", "val": "validation"}[split]
    return (
        Path(os.environ.get("DATAPATH", "/data"))
        / f"challenge_{ext_split}_scenario_ids.txt"
    )


@app.command()
def generate(
    workspace: Path, run_hash: str, split: Mode, workers: int = 4, batch_size: int = 16
):
    """
    Export pytorch predictions to folder of numpy files
    """
    from konductor.trainer.init import cli_init_config
    from utils.export_torch import initialize, run_export

    cli_args = Namespace(
        workspace=workspace,
        run_hash=run_hash,
        split=split,
        workers=workers,
        batch_size=batch_size,
        config_file=None,
    )

    exp_cfg = cli_init_config(cli_args)

    with get_id_path(split.name).open("r", encoding="utf-8") as f:
        scenario_ids = set([l.strip() for l in f.readlines()])

    model, dataloader = initialize(exp_cfg, cli_args)
    pred_path = workspace / run_hash / f"{split.name}_blobs"
    gt_path = workspace / f"{split.name}_ground_truth"

    pred_path.mkdir(exist_ok=True)
    gt_path.mkdir(exist_ok=True)

    run_export(model, dataloader, scenario_ids, pred_path, gt_path, batch_size)


@app.command()
def evaluate(
    workspace: Path,
    run_hash: str,
    split: Mode,
    visualize: bool = False,
):
    """
    Use waymo evaluation code for calculating IOU/AUC requires exported pytorch predictions
    """
    from utils.export_tf import evaluate_methods

    pred_path = workspace / run_hash / f"{split}_blobs"
    split_ = {"test": "testing", "val": "validation"}[split.name]
    pt_eval, tf_eval = evaluate_methods(
        get_id_path(split.name), pred_path, split_, visualize
    )
    meta = Metadata.from_yaml(workspace / run_hash / "metadata.yaml")

    with closing(get_db(workspace / "waymo_eval.db")) as con:
        cur = con.cursor()
        update_eval_db(cur, run_hash, meta.epoch, pt_eval)
        update_eval_db(cur, run_hash, meta.epoch, tf_eval)
        con.commit()


@app.command()
def export(workspace: Path, run_hash: str, split: Mode):
    """
    Export predictions
    """
    from utils.export_tf import export_evaluation

    export_evaluation(workspace / run_hash / f"{split.name}_blobs")


if __name__ == "__main__":
    app()
