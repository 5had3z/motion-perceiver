"""Export pytorch predictions to numpy file
to then import to waymo"""

from contextlib import closing
import enum
import os
from pathlib import Path
from shutil import rmtree
import time
from typing_extensions import Annotated

import typer
from colorama import Fore, Style

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
    workspace: Path,
    run_hash: str,
    split: Annotated[Mode, typer.Option()],
    workers: Annotated[int, typer.Option()] = 4,
    batch_size: Annotated[int, typer.Option()] = 16,
):
    """
    Export pytorch predictions to folder of numpy files
    """
    from konductor.trainer.init import get_experiment_cfg, add_workers_to_dataloader
    from utils.export_torch import initialize, run_export

    exp_cfg = get_experiment_cfg(workspace, None, run_hash)
    add_workers_to_dataloader(exp_cfg, workers)

    with get_id_path(split.name).open("r", encoding="utf-8") as f:
        scenario_ids = set([l.strip() for l in f.readlines()])

    model, dataloader = initialize(exp_cfg, split.name)
    pred_path = workspace / run_hash / f"{split.name}_blobs"
    gt_path = workspace / f"{split.name}_ground_truth"

    pred_path.mkdir(exist_ok=True)
    gt_path.mkdir(exist_ok=True)

    run_export(model, dataloader, scenario_ids, pred_path, gt_path, batch_size)


@app.command()
def evaluate(
    workspace: Path,
    run_hash: str,
    split: Annotated[Mode, typer.Option()],
    visualize: Annotated[bool, typer.Option()] = False,
):
    """
    Use waymo evaluation code for calculating IOU/AUC requires exported pytorch predictions
    """
    from utils.export_tf import evaluate_methods

    pred_path = workspace / run_hash / f"{split}_blobs"
    pt_eval, tf_eval = evaluate_methods(
        get_id_path(split.name), pred_path, split, visualize
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
def export(workspace: Path, run_hash: str, split: Annotated[Mode, typer.Option()]):
    """Export predictions for waymo eval server"""
    from utils.export_tf import export_evaluation

    export_evaluation(workspace / run_hash / f"{split.name}_blobs")


@app.command()
def clean(exp_path: Path):
    """Remove generated evaluation files from experiment"""
    pred_gen = [f"{s.name}_blobs" for s in Mode]
    for item in pred_gen:
        target = exp_path / item
        if not target.exists():
            continue
        elif target.is_dir():
            rmtree(target)
        elif target.is_file():
            target.unlink()


@app.command()
def clean_all(workspace: Path):
    """Remove generated evaluation files from all experiments"""
    for f in workspace.iterdir():
        if f.is_dir():
            clean(f)


@app.command()
def auto_evaluate(
    workspace: Path, _clean: Annotated[bool, typer.Option("--clean")] = True
):
    """
    Automatically perform val evaluation over experiments.
    Metadata is first recreated so we can determine what experiments are "out of date" or are
    missing entries.
    If "out of date" or missing an entry then we can run waymo evaluation on that experiment.
    """
    update_metadata(workspace)
    need_update = find_outdated_runs(workspace)
    print(f"{len(need_update)} experiments to update: {need_update}")

    emph = lambda x, c: f"{c+Style.BRIGHT}{x}{Style.RESET_ALL}"

    stime = time.perf_counter()
    for idx, run_hash in enumerate(need_update):
        try:
            print(emph(f"Running {run_hash}", Fore.BLUE))
            generate(workspace, run_hash, Mode.val)
            evaluate(workspace, run_hash, Mode.val)
            if _clean:
                clean(workspace / run_hash)
            print(
                emph(
                    f"Updated {idx+1}/{len(need_update)} Experiments"
                    f", elapsed {time.perf_counter()-stime}s",
                    Fore.BLUE,
                )
            )
        except Exception as err:
            print(emph(f"Skpping {run_hash} with error: {err}", Fore.RED))


if __name__ == "__main__":
    app()
