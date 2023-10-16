#!/usr/bin/env python3

"""Show the output of the dataloader
to validate its doing the right thing"""
from typing import Dict
from pathlib import Path
import os

from torch import Tensor, inference_mode
from konductor.data import ModuleInitConfig, Split
from konductor.data.dali import DaliLoaderConfig
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from src.dataset.common import MotionDatasetConfig
from src.dataset.interaction import InteractionConfig
from src.dataset.waymo import WaymoDatasetConfig
from src.dataset.eth_ucy import ETHUCYDatasetConfig
from src.dataset.sdd import SDDDatasetConfig
import src.dataset.visualisation as dv
import src.dataset.utils as du
from utils.visual import reverse_image_transforms


def run_viz(loader: DALIGenericIterator, config: MotionDatasetConfig) -> None:
    dump_dir = Path("viz_data")
    dump_dir.mkdir(exist_ok=True)
    os.chdir(dump_dir)

    for data in loader:
        data: Dict[str, Tensor] = data[0]  # remove list dim
        # dv.roadgraph(data["roadgraph"], data["roadgraph_valid"])
        # dv.roadmap(data["roadmap"])
        # dv.optical_flow(data["flow"])
        if "roadmap" in data and data["roadmap"].shape[1] == 1:
            dv.roadmap_and_occupancy(
                data["roadmap"],
                data["heatmap"],
                data.get("signals", None),
                roi_scale=config.occupancy_roi,
            )
        if "roadmap" in data and data["roadmap"].shape[1] == 3:
            data["roadmap"] = reverse_image_transforms(data["roadmap"])
            dv.context_image_occupancy(data["roadmap"], data["heatmap"])
        dv.scatterplot_sequence(data, 7)
        # dv.occupancy_from_current_pose(data)

        break


@inference_mode()
def main():
    batch_size = 8
    datacfg = WaymoDatasetConfig(
        train_loader=DaliLoaderConfig(batch_size=batch_size),
        val_loader=DaliLoaderConfig(
            batch_size=batch_size,
            # augmentations=[ModuleInitConfig("center", {})],
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
        heatmap_time=list(range(0, 77 // 2)),
        # random_heatmap_count=0,
        # random_heatmap_minmax=(0, 60),
        # signal_features=True,
        occupancy_roi=1,
        # only_vehicles=True,
        # flow_mask=True,
        # velocity_norm=4.0,
        time_stride=2,
    )
    dataloader: DALIGenericIterator = datacfg.get_dataloader(Split.VAL)
    run_viz(dataloader, datacfg)
    # du.velocity_distribution(dataloader, batch_size * 10)


if __name__ == "__main__":
    main()
