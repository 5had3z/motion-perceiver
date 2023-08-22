#!/usr/bin/env python3

"""Show the output of the dataloader
to validate its doing the right thing"""
from typing import Dict

import torch
from torch import Tensor, inference_mode
from konductor.data import ModuleInitConfig, Mode, get_dataloader
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from src.dataset.common import MotionDatasetConfig
from src.dataset.interaction import InteractionConfig
from src.dataset.waymo import WaymoDatasetConfig
from src.dataset.eth_ucy import ETHUCYDatasetConfig
from src.dataset.sdd import SDDDatasetConfig
import src.dataset.visualisation as dv
import src.dataset.utils as du


def run_viz(loader: DALIGenericIterator, config: MotionDatasetConfig) -> None:
    for data in loader:
        data: Dict[str, Tensor] = data[0]  # remove list dim
        if "roadmap" not in data:
            data["roadmap"] = torch.zeros((data["occupancy"].shape[0], 1, 100, 100))
        # dv.roadgraph(data["roadgraph"], data["roadgraph_valid"])
        # dv.roadmap(data["roadmap"])
        # dv.optical_flow(data["flow"])
        dv.roadmap_and_occupancy(
            data["roadmap"],
            data["occupancy"],
            data.get("signals", None),
            roi_scale=config.occupancy_roi,
        )
        dv.scatterplot_sequence(data, 7)
        # dv.occupancy_from_current_pose(data)

        break


@inference_mode()
def main():
    batch_size = 4
    datacfg = SDDDatasetConfig(
        train_loader=ModuleInitConfig(type="dali", args={"batch_size": batch_size}),
        val_loader=ModuleInitConfig(
            type="dali",
            args={
                "batch_size": batch_size,
                "augmentations": [ModuleInitConfig("center", {})],
            },
        ),
        # withheld="eth",
        full_sequence=True,
        map_normalize=30.0,
        occupancy_size=256,
        filter_future=True,
        roadmap_size=256,
        roadmap=True,
        # use_sdc_frame=True,
        # waymo_eval_frame=True,
        heatmap_time=list(range(0, 20)),
        # random_heatmap_count=0,
        # random_heatmap_minmax=(0, 60),
        # signal_features=False,
        # occupancy_roi=0.5,
        # only_vehicles=True,
        # flow_mask=True,
        # velocity_norm=4.0,
        time_stride=1,
    )
    dataloader: DALIGenericIterator = get_dataloader(datacfg, Mode.val)
    run_viz(dataloader, datacfg)
    # du.velocity_distribution(dataloader, batch_size * 10)


if __name__ == "__main__":
    main()
