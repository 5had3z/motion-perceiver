from pathlib import Path
from typing import List

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
    ]
)


@callback(
    Output("left-select", "options"),
    Output("right-select", "options"),
    Input("root-dir", "data"),
)
def init_exp(root_dir: str):
    if len(EXPERIMENTS) == 0:
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
