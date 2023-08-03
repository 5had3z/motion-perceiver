#!/usr/bin/env python3

"""Show the output of the dataloader
to validate its doing the right thing"""
from typing import Dict

from torch import Tensor, inference_mode
from konductor.data import ModuleInitConfig, Mode, get_dataloader
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from src.dataset.common import MotionDatasetConfig
from src.dataset.interaction import InteractionConfig
from src.dataset.waymo import WaymoDatasetConfig
import src.dataset.visualisation as dv
import src.dataset.utils as du


def run_viz(loader: DALIGenericIterator, config: MotionDatasetConfig) -> None:
    for data in loader:
        data: Dict[str, Tensor] = data[0]  # remove list dim
        # dv.roadgraph(data["roadgraph"], data["roadgraph_valid"])
        # dv.roadmap(data["roadmap"])
        # dv.optical_flow(data["flow"])
        dv.roadmap_and_occupancy(
            data["roadmap"],
            data["heatmap"],
            data.get("signals", None),
            roi_scale=config.occupancy_roi,
        )
        # dv.scatterplot_sequence(data, 10 // config.time_stride)
        # dv.occupancy_from_current_pose(data)

        break


@inference_mode()
def main():
    batch_size = 8
    datacfg = InteractionConfig(
        train_loader=ModuleInitConfig(type="dali", args={"batch_size": batch_size}),
        val_loader=ModuleInitConfig(
            type="dali",
            args={"batch_size": batch_size, "augmentations": {"random_rotate": {}}},
        ),
        full_sequence=True,
        map_normalize=80.0,
        occupancy_size=256,
        filter_future=True,
        roadmap_size=256,
        roadmap=True,
        # use_sdc_frame=True,
        # waymo_eval_frame=True,
        heatmap_time=list(range(0, 40, 10)),
        # random_heatmap_count=0,
        # random_heatmap_minmax=(0, 60),
        # signal_features=False,
        occupancy_roi=0.5,
        # only_vehicles=True,
        flow_mask=True,
        velocity_norm=4.0,
        time_stride=1,
    )
    dataloader: DALIGenericIterator = get_dataloader(datacfg, Mode.val)
    run_viz(dataloader, datacfg)
    # du.velocity_distribution(dataloader, batch_size * 10)


if __name__ == "__main__":
    main()
