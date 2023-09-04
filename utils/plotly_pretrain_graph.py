from pathlib import Path
from typing import Dict, List

import dash_cytoscape as cyto
import yaml
from dash import Input, Output, callback, html
from konductor.metadata.manager import Metadata
from konductor.init import ExperimentInitConfig
from konductor.webserver.utils import Experiment, fill_experiments

EXPERIMENTS: List[Experiment] = []

layout = html.Div(
    [
        html.H2("Pretrained Depenency Graph"),
        html.Div(
            "If the experiment is a parent or child in a pretraining dependency graph, it will show up here"
        ),
        cyto.Cytoscape(
            id="cytoscape",
            elements=[
                {
                    "data": {"id": "dummy", "label": "No Dependencies"},
                    "position": {"x": 75, "y": 75},
                },
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

    should_add = {}
    potentials = {}

    for e in EXPERIMENTS:
        exp_id = str(e.root.name)
        exp_meta = Metadata.from_yaml(e.root / "metadata.yaml")
        potentials[exp_id] = {"data": {"id": exp_id, "label": exp_meta.brief}}

        # Initialise potential node as false
        if exp_id not in should_add:
            should_add[exp_id] = False

        cfg_path = e.root / "train_config.yml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        config_dict["work_dir"] = e.root
        config = ExperimentInitConfig.from_yaml(config_dict)

        # Add edge if it is pretrained
        if "pretrained" in config.model[0].args:
            source = config.model[0].args["pretrained"].split(".")[0]
            elements_data.append({"data": {"source": source, "target": exp_id}})

            # Mark node as true
            should_add[exp_id] = True
            should_add[source] = True

    for id, data in potentials.items():
        if should_add[id]:
            elements_data.append(data)

    return elements_data
