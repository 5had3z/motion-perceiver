#!/usr/bin/env python3
"""Scripts for converting various datasets into tfrecord format"""
import typer

from utils.dataset.make_eth_ucy import app as euro_app
from utils.dataset.make_interaction import app as inter_app
from utils.dataset.trajdata2tfrec import app as traj_app

app = typer.Typer()
app.add_typer(euro_app, name="eth-ucy")
app.add_typer(inter_app, name="interaction")
app.add_typer(traj_app, name="trajdata")

if __name__ == "__main__":
    app()
