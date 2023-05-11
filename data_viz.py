"""Show the output of the dataloader
to validate its doing the right thing"""
from typing import Dict

from torch import Tensor
from konductor.modules.data import ModuleInitConfig, Mode, get_dataloader

from src.dataset.waymo import WaymoDatasetConfig
import src.dataset.visualisation as mv


def main():
    batch_size = 4
    waymo = WaymoDatasetConfig(
        train_loader=ModuleInitConfig(type="dali", args={"batch_size": batch_size}),
        val_loader=ModuleInitConfig(type="dali", args={"batch_size": batch_size}),
        full_sequence=True,
        map_normalize=80.0,
        occupancy_size=256,
        filter_future=True,
        roadmap_size=256,
        roadmap=True,
        use_sdc_frame=True,
        waymo_eval_frame=True,
        heatmap_time=list(range(0, 61, 10)),
        random_heatmap_count=0,
        random_heatmap_minmax=(0, 60),
        signal_features=True,
        occupancy_roi=0.5,
        only_vehicles=True,
        flow_mask=True,
    )
    dataloader = get_dataloader(waymo, Mode.val)

    for data in dataloader:
        data: Dict[str, Tensor] = data[0]  # remove list dim
        # mv.roadgraph(data["roadgraph"], data["roadgraph_valid"])
        # mv.roadmap(data["roadmap"])
        mv.optical_flow(data["flow"])
        # mv.roadmap_and_occupancy(
        #     data["roadmap"],
        #     data["heatmap"],
        #     data["signals"],
        #     roi_scale=waymo.occupancy_roi,
        # )
        # mv.sequence(data)
        # mv.occupancy_from_current_pose(data)
        break


if __name__ == "__main__":
    main()
