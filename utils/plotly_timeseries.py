from pathlib import Path
from typing import List
import difflib

import pandas as pd
from dash import html, dcc, Input, Output, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from konductor.webserver.utils import Experiment, fill_experiments

from .plot_utils import gather_experiment_time_performance

EXPERIMENTS: List[Experiment] = []

layout = html.Div(
    children=[
        html.H1(children="Motion Perceiver"),
        dbc.Row(
            [
                dcc.Dropdown(id="dd-metric", options=["IoU", "AUC"]),
                dcc.Graph(id="timeseries-graph", selectedData={}),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Dropdown(id="left-select"),
                        dcc.Textarea(
                            id="left-comp",
                            readOnly=True,
                            style={"width": "100%", "height": 300},
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        html.H4("Config Difference", style={"text-align": "center"}),
                        dcc.Textarea(
                            id="diff-comp",
                            readOnly=True,
                            style={"width": "100%", "height": 300},
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        dcc.Dropdown(id="right-select"),
                        dcc.Textarea(
                            id="right-comp",
                            readOnly=True,
                            style={"width": "100%", "height": 300},
                        ),
                    ]
                ),
            ]
        ),
    ]
)


@callback(
    Output("left-select", "options"),
    Output("right-select", "options"),
    Input("root-dir", "data"),
)
def init_exp(root_dir: str):
    fill_experiments(Path(root_dir), EXPERIMENTS)
    opts = [e.name for e in EXPERIMENTS]
    return opts, opts


@callback(
    Output("timeseries-graph", "figure"),
    Input("dd-metric", "value"),
)
def update_graph(metric: str):
    if not metric:
        raise PreventUpdate

    exps: List[pd.Series] = gather_experiment_time_performance(
        EXPERIMENTS, "val", metric
    )
    if len(exps) == 0:
        raise PreventUpdate

    fig = go.Figure()
    for exp in exps:
        fig.add_trace(
            go.Scatter(x=exp.index, y=exp.values, mode="lines", name=exp.name)
        )

    return fig


@callback(Output("left-comp", "value"), Input("left-select", "value"))
def update_left(exp_name):
    if not exp_name:
        raise PreventUpdate
    exp = next(x for x in EXPERIMENTS if x.name == exp_name)
    with open(exp.root / "train_config.yml", "r", encoding="utf-8") as f:
        s = f.read()
    return s


@callback(Output("right-comp", "value"), Input("right-select", "value"))
def update_right(exp_name):
    if not exp_name:
        raise PreventUpdate
    exp = next(x for x in EXPERIMENTS if x.name == exp_name)
    with open(exp.root / "train_config.yml", "r", encoding="utf-8") as f:
        s = f.read()
    return s


@callback(
    Output("diff-comp", "value"),
    Input("left-select", "value"),
    Input("right-select", "value"),
)
def diff_files(left_file, right_file):
    if not all([left_file, right_file]):
        raise PreventUpdate

    exp = next(x for x in EXPERIMENTS if x.name == left_file)
    with open(exp.root / "train_config.yml", "r", encoding="utf-8") as f:
        left = f.readlines()

    exp = next(x for x in EXPERIMENTS if x.name == right_file)
    with open(exp.root / "train_config.yml", "r", encoding="utf-8") as f:
        right = f.readlines()

    diff = difflib.unified_diff(left, right, fromfile=left_file, tofile=right_file)

    return "".join([d for d in diff])
