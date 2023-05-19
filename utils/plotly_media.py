from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image
from dash import html, dcc, Input, Output, callback, ALL, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from konductor.webserver.utils import Experiment, fill_experiments

EXPERIMENTS: List[Experiment] = []

layout = html.Div(
    children=[
        html.H1(children="Media Viewer"),
        dbc.Row(
            [
                dbc.Col(dcc.Dropdown(id="md-experiment")),
                dbc.Col(dcc.Dropdown(id="md-type", options=["occupancy", "flow"])),
            ]
        ),
        dbc.Row(
            html.Video(
                id="md-video",
                controls=True,
                height="500px",
            )
        ),
        html.Div(id="md-thumbnails"),
    ]
)


@callback(
    Output("md-experiment", "options"),
    Input("root-dir", "data"),
)
def init_exp(root_dir: str):
    if len(EXPERIMENTS) == 0:
        fill_experiments(Path(root_dir), EXPERIMENTS)
    return [e.name for e in EXPERIMENTS]


def get_thumbnail(path: Path) -> np.ndarray:
    """Reads video and returns first frame"""
    vid = cv2.VideoCapture(str(path))
    _, im = vid.read()
    vid.release()
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def add_thumbnail_to_grid(grid: List[dbc.Col], video_path: Path, n_col: int = 3):
    """Adds thumbnail to grid with a width of n_col (infinite rows)"""
    img = get_thumbnail(video_path)

    grid.append(
        dbc.Col(
            html.Img(
                id={"type": "vid-thumbnail", "index": f"{video_path.stem}"},
                src=Image.fromarray(img).resize((300, 300)),
                alt=video_path.stem,
            ),
            style={"text-align": "center"},
        )
    )


@callback(
    Output("md-thumbnails", "children"),
    Input("root-dir", "data"),
    Input("md-experiment", "value"),
    Input("md-type", "value"),
)
def update_thumbnails(root_dir: str, experiment: str, media: str):
    """TODO make this more adaptable to the width of the window"""
    if not all([experiment, root_dir, media]):
        raise PreventUpdate

    exp = next(e for e in EXPERIMENTS if e.name == experiment)
    children = []
    for video_path in exp.root.glob(f"**/{media}/*.webm"):
        add_thumbnail_to_grid(children, video_path)

    return dbc.Row(children)


@callback(
    Output("md-video", "src"),
    Input({"type": "vid-thumbnail", "index": ALL}, "n_clicks"),
    Input("md-experiment", "value"),
    Input("content-url", "data"),
)
def select_video(nclicks, experiment, url):
    if not experiment or len(nclicks) == 0 or ctx.triggered_id is None:
        raise PreventUpdate

    exp = next(e for e in EXPERIMENTS if e.name == experiment)
    vidpath = next(exp.root.glob(f"**/{ctx.triggered_id['index']}.webm"))
    relpath = Path(exp.root.stem) / vidpath.relative_to(exp.root)
    print(relpath)
    return f"{url}/{relpath}"
