from argparse import ArgumentParser
from subprocess import Popen

import dash
from konductor.webserver.app import app, get_basic_layout, add_base_args
from utils import plotly_timeseries, plotly_media

dash.register_page(
    "Timeseries", path="/timeseries-performance", layout=plotly_timeseries.layout
)

dash.register_page("Media", path="/media", layout=plotly_media.layout)

if __name__ == "__main__":
    parser = ArgumentParser()
    add_base_args(parser)
    content_port = 8000
    content_url = f"http://localhost:{content_port}"
    app.layout = get_basic_layout(str(parser.parse_args().workspace), content_url)

    proc = Popen(
        f"python3 -m http.server {content_port} --directory {parser.parse_args().workspace}",
        shell=True,
    )

    app.run(debug=True)
    proc.terminate()
