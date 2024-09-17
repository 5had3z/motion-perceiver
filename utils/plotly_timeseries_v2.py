import re
import sqlite3
from contextlib import closing
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html
from dash.exceptions import PreventUpdate

_WAYPOINT_COL_RE = re.compile(r"\b[a-zA-Z]+_\d+\b")

layout = html.Div(
    children=[
        dbc.Row(
            [
                html.H3("Eval Accuracy Over Time"),
                dcc.Dropdown(id="ts2-metric", options=["iou", "auc", "epe"]),
                dcc.Graph(id="ts2-graph", selectedData={}),
            ]
        ),
        dbc.Row(
            [
                html.H3("Evaluation Metrics"),
                dash_table.DataTable(id="ts2-table", sort_action="native"),
            ]
        ),
    ]
)


def get_performance_data(root: Path):
    with closing(sqlite3.connect(root / "results.db")) as con:
        perf = pd.read_sql_query("SELECT * FROM waypoints", con, index_col="hash")
        meta = pd.read_sql_query("SELECT * FROM metadata", con, index_col="hash")

    # Join metadata to performance data (and drop duplicates)
    duplicate_cols = ["epoch"]
    perf = perf.join(meta.drop(columns=duplicate_cols))

    return perf


@callback(
    Output("ts2-graph", "figure"),
    Input("ts2-metric", "value"),
    Input("root-dir", "data"),
    prevent_initial_call=True,
)
def update_graph(metric: str, root: str):
    if not all([metric, root]):
        raise PreventUpdate

    df = get_performance_data(Path(root))

    df.set_index("desc", inplace=True)
    df.drop(
        columns=[c for c in df.columns if not c.startswith(metric) or c == "desc"],
        inplace=True,
    )
    df.drop(columns=f"{metric}_mean", inplace=True)  # Drop mean metric

    fig = go.Figure()
    for index, values in df.iterrows():
        values.index = np.array([int(i.split("_")[-1]) for i in values.index])
        fig.add_trace(
            go.Scatter(x=values.index, y=values.values, mode="lines", name=index)
        )

    return fig


@callback(
    Output("ts2-table", "data"),
    Output("ts2-table", "columns"),
    Input("root-dir", "data"),
)
def update_table(root: str):
    if not root:
        raise PreventUpdate

    perf = get_performance_data(Path(root))

    # remove any stat_ts keys from the table
    filter_keys = [c for c in perf.columns if _WAYPOINT_COL_RE.match(c)]
    filter_keys.append("iteration")  # No need for iteration either
    perf.drop(columns=filter_keys, inplace=True)

    labels = [{"name": i, "id": i} for i in ["ts", "epoch", "desc"]]

    for l in ["iou", "auc", "epe"]:
        labels.append({"name": l, "id": f"{l}_mean"})

    return perf.to_dict("records"), labels
