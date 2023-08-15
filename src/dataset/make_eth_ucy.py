"""
Converts eth_ucy dataset into tfrecords that are slices of sequences for
easy and efficient dataloading with DALI.
If the stride of the slicing algorithm is equal to the sequence length,
then these will be non-overlapping.
Simple interpolation will be used to calculate velocity of the agents, the
direction of velocity will inform heading as most people walk the way they face.
"""
from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def max_agents_per_slice(path: Path, length: int):
    """Find the maximum number of agents for a given sequence length
    This is usefull to figure out the dimensions of the tfrecords."""


@app.command()
def build_dataset(path: Path, length: int, stride: int, max_agents: int):
    """Build the dataset with sequence length and dimension max_agents, sample at stride"""


if __name__ == "__main__":
    app()
