#!/usr/bin/env python3

import dash
from konductor.webserver.app import cliapp

from utils import plotly_pretrain_graph, plotly_timeseries, plotly_timeseries_v2

dash.register_page(
    "Timeseries", path="/timeseries-performance", layout=plotly_timeseries.layout
)

dash.register_page(
    "Timeseries2", path="/timeseries-2", layout=plotly_timeseries_v2.layout
)

dash.register_page(
    "TrainGraph", path="/train-graph", layout=plotly_pretrain_graph.layout
)

if __name__ == "__main__":
    cliapp()
