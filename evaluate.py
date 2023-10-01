"""Export pytorch predictions to numpy file
to then import to waymo"""

import os
import time
from contextlib import closing
from pathlib import Path
from shutil import rmtree

import typer
from colorama import Fore, Style
from konductor.init import Split
from typing_extensions import Annotated

from utils.eval_data import (
    Metadata,
    find_outdated_runs,
    get_db,
    upsert_eval,
    upsert_metadata,
)

app = typer.Typer()


def get_id_path(split: Split):
    """Get path to ID subset file of split"""
    ext_split = {Split.TEST: "testing", Split.VAL: "validation"}[split]
    return (
        Path(os.environ.get("DATAPATH", "/data"))
        / "waymo-motion"
        / "tf_example"
        / f"challenge_{ext_split}_scenario_ids.txt"
    )


@app.command()
def generate(
    run_path: Path,
    split: Annotated[Split, typer.Option()],
    workers: Annotated[int, typer.Option()] = 4,
    batch_size: Annotated[int, typer.Option()] = 16,
):
    """
    Export pytorch predictions to folder of numpy files
    """
    from konductor.data import get_dataloader
    from konductor.trainer.init import get_experiment_cfg

    from utils.eval_common import initialize
    from utils.export_torch import run_export

    exp_cfg = get_experiment_cfg(run_path.parent, None, run_path.name)
    exp_cfg.set_workers(workers)
    exp_cfg.set_batch_size(batch_size, split)

    with open(get_id_path(split), "r", encoding="utf-8") as f:
        scenario_ids = {l.strip() for l in f.readlines()}

    model, dataset = initialize(exp_cfg, eval_waypoints=True)
    dataloader = get_dataloader(dataset, split)

    split_name = split.name.lower()
    pred_path = run_path / f"{split_name}_blobs"
    gt_path = run_path.parent / f"{split_name}_ground_truth"

    pred_path.mkdir(exist_ok=True)
    gt_path.mkdir(exist_ok=True)

    run_export(model, dataloader, scenario_ids, pred_path, gt_path, batch_size)


@app.command()
def evaluate(
    run_path: Path,
    split: Annotated[Split, typer.Option()],
    visualize: Annotated[bool, typer.Option()] = False,
    save: Annotated[bool, typer.Option(help="Save result to db")] = True,
):
    """
    Use waymo evaluation code for calculating IOU/AUC requires exported pytorch predictions
    """
    from functools import partial

    from utils.export_tf import _get_validation_and_prediction, evaluate_methods

    eval_fn = partial(_get_validation_and_prediction, visualize=visualize)

    pred_path = run_path / f"{split.name.lower()}_blobs"
    pt_eval, tf_eval = evaluate_methods(get_id_path(split), pred_path, split, eval_fn)

    if not save:
        print(str(pt_eval), str(tf_eval))
        return

    meta = Metadata.from_yaml(run_path / "metadata.yaml")
    with closing(get_db(run_path.parent)) as con:
        cur = con.cursor()
        upsert_eval(cur, run_path.name, meta.epoch, pt_eval)
        upsert_eval(cur, run_path.name, meta.epoch, tf_eval)
        con.commit()


@app.command()
def waypoint_evaluate(
    run_path: Path,
    split: Annotated[Split, typer.Option()],
    save: Annotated[bool, typer.Option(help="Save result to db")] = True,
):
    """Evaluate waymo motion and collect waypoint accuarcy as well as mean"""
    from utils.eval_data import (
        create_waypoints_table,
        metric_data_list_to_dict,
        upsert_waypoints,
    )
    from utils.export_tf import _evaluate_timepoints_and_mean, evaluate_methods

    pred_path = run_path / f"{split}_blobs"
    metrics = evaluate_methods(
        get_id_path(split), pred_path, split, _evaluate_timepoints_and_mean
    )

    if not save:
        print("\n".join(str(m) for m in metrics))
        return

    data_dict = metric_data_list_to_dict(metrics)
    meta = Metadata.from_yaml(run_path / "metadata.yaml")

    with closing(get_db(run_path.parent)) as con:
        cur = con.cursor()
        create_waypoints_table(cur)
        upsert_waypoints(cur, run_path.name, meta, data_dict)
        con.commit()


@app.command()
def torch_evaluate(
    run_path: Path,
    split: Annotated[Split, typer.Option()] = Split.VAL,
    workers: Annotated[int, typer.Option()] = 4,
    batch_size: Annotated[int, typer.Option()] = 16,
):
    """Run evaluation with pytorch code"""
    from konductor.data import get_dataloader_config
    from konductor.metadata.statistics.pq_writer import _ParquetWriter
    from konductor.trainer.init import get_experiment_cfg

    from src.statistics import Occupancy
    from utils.eval_common import initialize
    from utils.export_torch import run_eval

    exp_cfg = get_experiment_cfg(run_path.parent, None, run_path.name)
    exp_cfg.set_workers(workers)
    exp_cfg.set_batch_size(batch_size, split)

    tmp_root = Path(".cache")
    tmp_root.mkdir(exist_ok=True)
    writer = _ParquetWriter(tmp_root, "eval_occupancy")

    model, dataset = initialize(exp_cfg, eval_waypoints=True)

    perf_logger = {Occupancy(time_idxs=dataset.heatmap_time)}
    dataloader_cfg = get_dataloader_config(dataset, split)
    dataloader_cfg.drop_last = False
    dataloader = dataloader_cfg.get_instance()

    run_eval(model, dataloader, perf_logger)

    final_data = perf_logger.data()

    for k, v in final_data.items():
        print(f"{k}: {v.mean()}")

    writer.unlink()  # clean up


@app.command()
def update_metadata(workspace: Path):
    """Apply updates to the metadata table in the eval database"""

    def iterate_metadata():
        """Iterate over metadata files in workspace"""
        for run in workspace.iterdir():
            metapath = run / "metadata.yaml"
            if metapath.exists():
                yield metapath

    with closing(get_db(workspace)) as con:
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
    need_update = find_outdated_runs(workspace, "waypoints")
    print(f"{len(need_update)} experiments to update: {need_update}")

    emph = lambda x, c: f"{c+Style.BRIGHT}{x}{Style.RESET_ALL}"

    stime = time.perf_counter()
    for idx, run_hash in enumerate(need_update):
        if (workspace / run_hash / "NO_EVAL").exists():
            print(f"Skipping {run_hash} marked as NO EVAL")
            continue  # Skip experiment if marked no eval
        try:
            print(emph(f"Running {run_hash}", Fore.BLUE))
            generate(workspace, run_hash, Mode.val)
            waypoint_evaluate(workspace / run_hash, Mode.val)
            print(
                emph(
                    f"Updated {idx+1}/{len(need_update)} Experiments"
                    f", elapsed {time.perf_counter()-stime}s",
                    Fore.BLUE,
                )
            )
        except Exception as err:
            print(emph(f"Skpping {run_hash} with error: {err}", Fore.RED))
        if _clean:
            clean(workspace / run_hash)


if __name__ == "__main__":
    app()
