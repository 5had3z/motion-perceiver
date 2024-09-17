import sqlite3
from pathlib import Path
from typing import List

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html
from dash.exceptions import PreventUpdate
from konductor.webserver.utils import Experiment, fill_experiments

from .plot_utils import gather_experiment_time_performance

EXPERIMENTS: List[Experiment] = []

layout = html.Div(
    children=[
        dbc.Row(
            [
                html.H3("Accuracy Over Time (PyTorch)"),
                dcc.Dropdown(id="dd-metric", options=["IoU", "AUC"]),
                dcc.Graph(id="timeseries-graph", selectedData={}),
            ]
        ),
        dbc.Row(
            [
                html.H3("Waymo Evaluation Metrics"),
                dcc.Dropdown(id="dd-method", options=["pytorch", "tensorflow"]),
                dash_table.DataTable(id="dd-table", sort_action="native"),
            ]
        ),
    ]
)


@callback(
    Output("timeseries-graph", "figure"),
    Input("dd-metric", "value"),
    Input("root-dir", "data"),
)
def update_graph(metric: str, root_dir: str):
    if len(EXPERIMENTS) == 0:
        fill_experiments(Path(root_dir), EXPERIMENTS)

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


@callback(
    Output("dd-table", "data"),
    Output("dd-table", "columns"),
    Input("dd-method", "value"),
    Input("root-dir", "data"),
)
def update_table(table: str, root: str):
    if not all([table, root]):
        raise PreventUpdate

    perf_db = sqlite3.connect(Path(root) / "results.db")
    perf = pd.read_sql_query(f"SELECT * FROM {table}", perf_db, index_col="hash")
    meta = pd.read_sql_query("SELECT * FROM metadata", perf_db, index_col="hash")
    perf_db.close()

    perf = perf.join(meta.drop(columns="epoch"))

    return perf.to_dict("records"), [
        {"name": i, "id": i} for i in ["ts", "epoch", "desc", "iou", "auc"]
    ]
