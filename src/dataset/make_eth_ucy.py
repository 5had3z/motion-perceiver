"""
Converts eth_ucy dataset into tfrecords that are slices of sequences for
easy and efficient dataloading with DALI.
If the stride of the slicing algorithm is equal to the sequence length,
then these will be non-overlapping.
Simple interpolation will be used to calculate velocity of the agents, the
direction of velocity will inform heading as most people walk the way they face.
Data should be sourced from here (inline with nvlabs/trajdata):
https://github.com/StanfordASL/Trajectron-plus-plus/tree/master/experiments/pedestrians/raw/raw/all_data
"""
from pathlib import Path
from typing import List, Optional
from typing_extensions import Annotated

import numpy as np
import pandas as pd
import tensorflow as tf
import typer

app = typer.Typer()

from src.dataset.eth_ucy import SEQUENCE_LENGTH, SUBSETS, MAX_AGENTS

TXT2RECORD = {
    "biwi_eth": "eth",
    "biwi_hotel": "hotel",
    "crowds_zara01": "zara1",
    "crowds_zara02": "zara2",
    "crowds_zara03": "zara3",
    "students001": "students1",
    "students003": "students3",
    "uni_examples": "uni",
}

assert (
    set(TXT2RECORD.values()) == SUBSETS
), "Mismatch between defined subsets and txt2record mapping"


def get_filenames(root: Path):
    filenames = list(root.iterdir())
    assert check_all_present(filenames)
    return filenames


def check_all_present(filenames: List[Path]):
    present = set(f.stem for f in filenames)
    expected = set(f for f in TXT2RECORD)
    unexpected = present.difference(expected)
    if len(unexpected) > 0:
        print(f"Unexpected files: {unexpected}")
    missing = expected.difference(present)
    if len(missing) > 0:
        print(f"Missing files: {missing}")
    return all(len(s) == 0 for s in [unexpected, missing])


def format_dataframe(df: pd.DataFrame):
    """
    Format up dataframe to {ts: int, id: int, x: float, y: float}
    and normalize ts stride to 1 and begining from zero
    """
    # Add names to columns
    df.columns = ["ts", "id", "x", "y"]

    # Change TS to int and normalize to zero
    df["ts"] = df["ts"].astype("int") // 10
    df["ts"] = df["ts"] - df["ts"].min()

    # Change ID to Int
    df["id"] = df["id"].astype("int")

    # Index by timestamp and sort
    df.set_index("ts")
    df.sort_index(axis="index", inplace=True)


@app.command()
def get_statistics(
    path: Path, length: Annotated[int, typer.Option()] = SEQUENCE_LENGTH
):
    """
    Run over the datsets with a given sequence length.
    and collect useful statistics such as:
        - Maximum number of agents in a scene
        - Maximum scene dimensions
    This data will be useful for the tfrecord creation
    and data normalisation.
    """
    filenames = get_filenames(path)

    max_agents = 0
    max_extent = 0
    for dataset in filenames:
        data = pd.read_csv(dataset, sep="\t", index_col=False, header=None)
        format_dataframe(data)
        max_extent = max(max_extent, data["x"].abs().max(), data["y"].abs().max())
        for start_ts in range(0, data["ts"].max(), length):
            seq = data[data["ts"].between(start_ts, start_ts + length - 1)]
            max_agents = max(max_agents, len(seq["id"].unique()))

    print(f"{max_extent=:.2f}, {max_agents=}")


def interpolate(data: np.ndarray, begin: int, end: int):
    first = data[begin - 1]
    last = data[end + 1]
    interp = np.linspace(first, last, end - begin + 3)
    return interp[1:-1]


def clean_data(valid: np.ndarray, x: np.ndarray, y: np.ndarray):
    """
    Insert repeat values at start and end, linear interpolate inbetween
    does not modify "valid" mask, most of this sanitation is to prevent
    erronious values in vx/vy/t/vt
    """
    valid = valid.copy()  # Make local copy

    # Extend repeat data outside of valid range
    first_valid = np.argmax(valid)
    x[:first_valid] = x[first_valid]
    y[:first_valid] = y[first_valid]
    valid[:first_valid] = 1

    last_valid = np.argmax(valid[::-1])
    x[last_valid:] = x[last_valid]
    y[last_valid:] = y[last_valid]
    valid[last_valid:] = 1

    # Linear Interpolate data inbetween
    while not valid.all():
        sidx = np.argmin(valid).item()  # Find first invalid
        eidx = np.argmax(valid[sidx:]).item() + sidx - 1  # Find last valid
        x[sidx : eidx + 1] = interpolate(x, sidx, eidx)
        y[sidx : eidx + 1] = interpolate(y, sidx, eidx)
        valid[sidx : eidx + 1] = 1


def velocity(data: np.ndarray) -> np.ndarray:
    vel = data[:, 1:] - data[:, :-1]
    vel = np.concatenate([vel, vel[:, [-1]]], axis=1)
    return vel


def make_tf_example(
    df: pd.DataFrame, scenario_id: str, length: int, max_agents: int
) -> tf.train.Example | None:
    """Make tensorflow example with centered coordinate system
    and interpolated t(headding), vx, vy,"""
    x = np.zeros((max_agents, length), dtype=np.float32)
    y = x.copy()
    valid = np.zeros((max_agents, length), dtype=np.int64)

    # Make local copy to not mutate outer timestamps
    df = df.copy()
    df["ts"] = df["ts"] - df["ts"].min()

    uids = df["id"].unique()

    if len(uids) < 2:
        return None

    # Fill out valid data
    for idx, id in enumerate(uids):
        agent = df[df["id"] == id]
        valid[idx, agent["ts"]] = 1
        x[idx, agent["ts"]] = agent["x"]
        y[idx, agent["ts"]] = agent["y"]

        clean_data(valid[idx], x[idx], y[idx])

    # Calculate interpolated values
    vx = velocity(x)
    vy = velocity(y)
    t = np.arctan2(vy, vx)
    vt = velocity(t)

    # Make tf_example
    tf_sample = tf.train.Example(
        features=tf.train.Features(
            # fmt: off
            feature={
                "x":            tf.train.Feature(float_list=tf.train.FloatList(value=x.flatten())),
                "y":            tf.train.Feature(float_list=tf.train.FloatList(value=y.flatten())),
                "t":            tf.train.Feature(float_list=tf.train.FloatList(value=t.flatten())),
                "vx":           tf.train.Feature(float_list=tf.train.FloatList(value=vx.flatten())),
                "vy":           tf.train.Feature(float_list=tf.train.FloatList(value=vy.flatten())),
                "vt":           tf.train.Feature(float_list=tf.train.FloatList(value=vt.flatten())),
                "valid":        tf.train.Feature(int64_list=tf.train.Int64List(value=valid.flatten())),
                "scenario_id":  tf.train.Feature(bytes_list=tf.train.BytesList(value=[scenario_id.encode("utf-8")])),
            }
            # fmt: on
        )
    )

    return tf_sample


def write_tfrecord(dest: Path, data: List[tf.train.Example]):
    with tf.io.TFRecordWriter(str(dest)) as writer:
        for sample in data:
            writer.write(sample.SerializeToString())


def create_tfrecord_dataset(
    filename: Path, dest: Path, stride: int, length: int, max_agents: int
):
    print(f"Building {filename.stem}")
    data = pd.read_csv(filename, sep="\t", index_col=False, header=None)
    format_dataframe(data)

    tf_samples: List[tf.train.Example] = []
    for start_ts in range(0, data["ts"].max(), stride):
        seq = data[data["ts"].between(start_ts, start_ts + length - 1)]
        scenario = f"{filename.stem}_{start_ts}"
        tf_sample = make_tf_example(seq, scenario, length, max_agents)
        if tf_sample is not None:
            tf_samples.append(tf_sample)

    print(f"Writing {filename.stem} with {len(tf_samples)} samples")
    write_tfrecord(dest, tf_samples)


@app.command()
def build(
    source: Path,
    dest: Annotated[Optional[Path], typer.Option()] = None,
    stride: Annotated[Optional[int], typer.Option()] = None,
    length: Annotated[int, typer.Option()] = SEQUENCE_LENGTH,
    max_agents: Annotated[int, typer.Option()] = MAX_AGENTS,
):
    """Build the dataset with sequence length and dimension max_agents, sample at stride"""
    if stride is None:
        stride = length

    if dest is None:
        dest = source.parent / f"{source.stem}_tfrecord"
    dest.mkdir(exist_ok=True)

    filenames = get_filenames(source)
    for dataset in filenames:
        tfrecord_file = (dest / TXT2RECORD[dataset.stem]).with_suffix(".tfrecord")
        create_tfrecord_dataset(dataset, tfrecord_file, stride, length, max_agents)


if __name__ == "__main__":
    app()
