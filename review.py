from argparse import ArgumentParser

import dash
from konductor.webserver.app import app, get_basic_layout, add_base_args
from utils import plotly_timeseries

dash.register_page(
    "Timeseries", path="/timeseries-performance", layout=plotly_timeseries.layout
)


if __name__ == "__main__":
    parser = ArgumentParser()
    add_base_args(parser)
    app.layout = get_basic_layout(str(parser.parse_args().root))

    app.run(debug=True)
