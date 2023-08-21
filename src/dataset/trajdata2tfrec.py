"""
Instead of dealing with each different dataset and its edge
cases (sanitation etc.), I can use trajdata to load batches,
then I can serialise into tfrecord format that I want.
"""
from pathlib import Path
from multiprocessing import cpu_count
from typing import Dict, List, Optional
from typing_extensions import Annotated

from konductor.utilities.pbar import LivePbar
import tensorflow as tf
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from trajdata import SceneBatch, AgentType, UnifiedDataset
from trajdata.visualization.vis import plot_scene_batch
import typer
import yaml

app = typer.Typer()


def get_subsets(name: str, root: Path) -> Dict[str, str]:
    if name == "sdd":
        dataset_map = {"sdd": str(root / "stanford_campus_dataset" / "annotations")}
    elif name == "eupeds":
        dataset_map = {
            f"eupeds_{d}": str(root / "eth_ucy")
            for d in ["eth", "hotel", "univ", "zara1", "zara2"]
        }
    else:
        raise NotImplementedError(name)

    return dataset_map


@app.command()
def build_cache(name: str, root: Path):
    data_dirs = get_subsets(name, root)
    dataset = UnifiedDataset(
        desired_data=list(data_dirs.keys()),
        rebuild_cache=True,
        rebuild_maps=True,
        num_workers=cpu_count(),
        verbose=True,
        data_dirs=data_dirs,
    )
    print(f"Total Data Samples: {len(dataset):,}")


@app.command()
def show(name: str, root: Path):
    data_dirs = get_subsets(name, root)

    dataset = UnifiedDataset(
        desired_data=list(data_dirs.keys()),
        centric="scene",
        desired_dt=0.1,
        history_sec=(2.8, 2.8),
        future_sec=(4.8, 4.8),
        only_types=[AgentType.PEDESTRIAN],
        num_workers=4,
        verbose=True,
        standardize_data=False,
        data_dirs=data_dirs,
    )

    print(f"# Data Samples: {len(dataset):,}")

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=12,
    )

    batch: SceneBatch
    with LivePbar(total=len(dataloader)):
        for batch in dataloader:
            plot_scene_batch(batch, batch_idx=0)


def get_dataset(
    split: str,
    data_dirs: Dict[str, str],
    history: float,
    future: float,
    period: float,
    max_agent: int | None = None,
) -> UnifiedDataset:
    """Get scene-centric dataset which isn't
    standardized with forwarded arguments"""
    dataset = UnifiedDataset(
        desired_data=[f"{d}-{split}" for d in data_dirs],
        centric="scene",
        desired_dt=period,
        history_sec=(history, history),
        future_sec=(future, future),
        num_workers=4,
        verbose=True,
        standardize_data=False,
        data_dirs=data_dirs,
        max_agent_num=max_agent,
    )
    return dataset


def make_tf_example(data: Dict[str, Tensor]) -> tf.train.Example:
    """Convert data dict into tf example"""
    cvt: Dict[str, tf.train.Feature] = {}
    for k, v in data.items():
        if isinstance(v, str):
            cvt[k] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[v.encode("utf-8")])
            )
        elif v.dtype == torch.float32:
            cvt[k] = tf.train.Feature(
                float_list=tf.train.FloatList(value=v.cpu().flatten().numpy())
            )
        elif v.dtype == torch.int64:
            cvt[k] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=v.cpu().flatten().numpy())
            )
        else:
            raise NotImplementedError(f"Can't serialise {v.dtype}")

    return tf.train.Example(features=tf.train.Features(feature=cvt))


def process_batch_to_tfexample(
    batch: SceneBatch, max_agents: int
) -> List[tf.train.Example]:
    """"""
    all_time = torch.cat([batch.agent_hist, batch.agent_fut], dim=2)
    bsz, n_agent, duration = all_time.shape[:3]
    assert n_agent <= max_agents, f"{n_agent=} exceeds {max_agents=}"

    data = {k: torch.zeros((bsz, max_agents, duration)) for k in ["x", "y", "vx", "vy"]}
    for idx, k in enumerate(data):
        data[k][:, :n_agent, :] = all_time[..., idx]

    # Add Headding/Theta
    data["t"] = torch.zeros_like(data["x"])
    data["t"][:, :n_agent, :] = torch.cat(
        [batch.agent_hist.heading, batch.agent_fut.heading], dim=2
    ).squeeze(-1)

    data["vt"] = torch.diff(data["t"], n=1, dim=-1)
    data["vt"] = torch.cat([data["vt"], data["vt"][..., [-1]]], dim=-1)

    data["type"] = torch.zeros((bsz, max_agents), dtype=torch.int64)
    data["type"][:, :n_agent] = batch.agent_type
    data["valid"] = torch.isfinite(data["x"]).to(torch.int64)

    for k in data:
        data[k][~torch.isfinite(data[k])] = 0

    data["scenario_id"]: List[str] = []
    for ts, name in zip(batch.scene_ts, batch.scene_ids):
        data["scenario_id"].append(f"{name}_{ts.item()}")

    tfexamples = []
    for bidx in range(bsz):
        tfexamples.append(make_tf_example({k: data[k][bidx] for k in data}))

    return tfexamples


def create_tfrecord(
    writer: tf.io.TFRecordWriter, dataloader: DataLoader, max_agents: int
):
    """Run over dataloader, converting yielded data to tf.Example
    and write to open tfrecordwriter"""
    with LivePbar(total=len(dataloader)) as pbar:
        for batch in dataloader:
            samples = process_batch_to_tfexample(batch, max_agents)
            for sample in samples:
                writer.write(sample.SerializeToString())
            pbar.update(1)


def write_metadata(
    path: Path,
    history: float,
    future: float,
    period: float,
    dataset: str,
    max_agents: int,
):
    """Write metadata of parameters used to create tfrecords"""
    metadata = {
        "dataset": dataset,
        "history_sec": history,
        "future_sec": future,
        "period_sec": period,
        "max_agents": max_agents,
    }

    with open(path, "w", encoding="utf-8") as w:
        yaml.safe_dump(metadata, w)


@app.command()
def make_tfrecords(
    name: str,
    root: Path,
    max_agents: Annotated[int, typer.Option()],
    dest: Annotated[Optional[Path], typer.Option()] = None,
    history: Annotated[float, typer.Option()] = 2.8,
    future: Annotated[float, typer.Option()] = 4.8,
    period: Annotated[float, typer.Option()] = 0.1,
):
    """Create train/val/test split tfrecords for
    dataset name, located in root and write output to dest"""
    if dest is None:
        dest = root.parent / f"{root.name}_tfrecord"
    dest.mkdir(exist_ok=True)

    data_dirs = get_subsets(name, root)

    for split in ["train", "val", "test"]:
        dataset = get_dataset(split, data_dirs, history, future, period, max_agents)

        print(f"# {split.capitalize()} Samples: {len(dataset):,}")

        dataloader = DataLoader(
            dataset,
            batch_size=16,
            collate_fn=dataset.get_collate_fn(),
            num_workers=cpu_count() // 2,
            shuffle=True,
        )

        tfrecord_file = dest / f"{name}_{split}.tfrecord"
        with tf.io.TFRecordWriter(str(tfrecord_file), options="ZLIB") as writer:
            create_tfrecord(writer, dataloader, max_agents)

        write_metadata(
            dest / "metadata.yaml", history, future, period, name, max_agents
        )


if __name__ == "__main__":
    with torch.inference_mode():
        app()
