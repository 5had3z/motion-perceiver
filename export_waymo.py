"""Export pytorch predictions to numpy file
to then import to waymo"""

from argparse import Namespace
from contextlib import closing
import enum
from pathlib import Path
import os
import time


import typer

from utils.eval_data import (
    upsert_eval,
    get_db,
    upsert_metadata,
    Metadata,
    find_outdated_runs,
)


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
        upsert_eval(cur, run_hash, meta.epoch, pt_eval)
        upsert_eval(cur, run_hash, meta.epoch, tf_eval)
        con.commit()


@app.command()
def update_metadata(workspace: Path):
    """Apply updates to the metadata table in the eval database"""

    def iterate_metadata():
        """Iterate over metadata files in workspace"""
        for run in workspace.iterdir():
            metapath = run / "metadata.yaml"
            if metapath.exists():
                yield metapath

    with closing(get_db(workspace / "waymo_eval.db")) as con:
        cur = con.cursor()
        for metaf in iterate_metadata():
            meta = Metadata.from_yaml(metaf)
            upsert_metadata(cur, str(metaf.parent.name), meta)
        con.commit()


@app.command()
def export(workspace: Path, run_hash: str, split: Mode):
    """
    Export predictions
    """
    from utils.export_tf import export_evaluation

    export_evaluation(workspace / run_hash / f"{split.name}_blobs")


@app.command()
def auto_evaluate(workspace: Path):
    """
    Automatically perform val evaluation over experiments.
    Metadata is first recreated so we can determine what experiments are "out of date" or are
    missing entries.
    If "out of date" or missing an entry then we can run waymo evaluation on that experiment.
    """
    update_metadata(workspace)
    need_updating = find_outdated_runs(workspace)
    print(f"{len(need_updating)} experiments to update: {need_updating}")

    stime = time.perf_counter()
    for idx, run_hash in enumerate(need_updating):
        generate(workspace, run_hash, Mode.val)
        evaluate(workspace, run_hash, Mode.val)
        print(
            f"Updated {idx}/{len(need_updating)} Experiments"
            f", elapsed {time.perf_counter()-stime}s"
        )


if __name__ == "__main__":
    app()
