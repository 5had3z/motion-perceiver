# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from argparse import ArgumentParser
from pathlib import Path
from typing import List

import pandas as pd
from dash import Dash, html, dcc, Input, Output
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from konductor.webserver.utils import get_experiments


from plot_utils import gather_experiment_time_performance

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

parser = ArgumentParser()
parser.add_argument("--root", type=Path, default=Path.cwd())
args = parser.parse_args()
experiments = get_experiments(args.root)

app.layout = html.Div(
    children=[
        html.H1(children="Motion Perceiver"),
        dcc.Dropdown(id="dd-metric", options=["IoU", "AUC"]),
        dcc.Graph(id="line-graph"),
    ]
)


@app.callback(
    Output("line-graph", "figure"),
    Input("dd-metric", "value"),
)
def update_graph(metric: str):
    if not metric:
        raise PreventUpdate

    exps: List[pd.Series] = gather_experiment_time_performance(
        experiments, "val", metric
    )
    if len(exps) == 0:
        raise PreventUpdate

    fig = go.Figure()
    for exp in exps:
        fig.add_trace(
            go.Scatter(x=exp.index, y=exp.values, mode="lines", name=exp.name)
        )

    return fig


if __name__ == "__main__":
    app.run(debug=True)
