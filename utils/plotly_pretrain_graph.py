from pathlib import Path
from typing import Dict, List

import dash_cytoscape as cyto
import yaml
from dash import Input, Output, callback, dcc, html
from konductor.metadata.manager import Metadata
from konductor.init import ExperimentInitConfig
from konductor.webserver.utils import Experiment, fill_experiments

EXPERIMENTS: List[Experiment] = []

layout = html.Div(
    [
        html.H1("Pretrained Depenency Graph"),
        cyto.Cytoscape(
            id="cytoscape",
            elements=[
                {
                    "data": {"id": "one", "label": "Node 1"},
                    "position": {"x": 75, "y": 75},
                },
                {
                    "data": {"id": "two", "label": "Node 2"},
                    "position": {"x": 200, "y": 200},
                },
                {"data": {"source": "one", "target": "two"}},
            ],
            layout={"name": "breadthfirst", "directed": True},
            style={"width": "100%", "height": "700px"},
        ),
    ]
)


@callback(Output("cytoscape", "elements"), Input("root-dir", "data"))
def init_exp(root_dir: str):
    if len(EXPERIMENTS) == 0:
        fill_experiments(Path(root_dir), EXPERIMENTS)

    elements_data: List[Dict[str, Dict[str, str]]] = []

    for e in EXPERIMENTS:
        exp_id = str(e.root.name)
        exp_meta = Metadata.from_yaml(e.root / "metadata.yaml")
        # Add Node
        elements_data.append({"data": {"id": exp_id, "label": exp_meta.brief}})

        cfg_path = e.root / "train_config.yml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        config_dict["work_dir"] = e.root
        config = ExperimentInitConfig.from_yaml(config_dict)

        # Add edge if it is pretrained
        if "pretrained" in config.model[0].args:
            source = config.model[0].args["pretrained"].split(".")[0]
            elements_data.append({"data": {"source": source, "target": exp_id}})

    return elements_data
