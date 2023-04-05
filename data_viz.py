"""Show the output of the dataloader
to validate its doing the right thing"""
from typing import Dict

from torch import Tensor
from konductor.modules.data import ModuleInitConfig, Mode, get_dataloader

from src.dataset.waymo import WaymoDatasetConfig
import src.dataset.visualisation as mv


def main():
    waymo = WaymoDatasetConfig(
        train_loader=ModuleInitConfig(type="dali", args={"batch_size": 4}),
        val_loader=ModuleInitConfig(type="dali", args={"batch_size": 4}),
        full_sequence=True,
        map_normalize=80.0,
        occupancy_size=256,
        filter_future=True,
        roadmap_size=256,
        roadmap=True,
        use_sdc_frame=True,
        waymo_eval_frame=True,
        heatmap_time=list(range(0, 91, 10)),
        signal_features=True,
        occupancy_roi=0.5,
        only_vehicles=True,
    )
    dataloader = get_dataloader(waymo, Mode.train)

    for data in dataloader:
        data: Dict[str, Tensor] = data[0]  # remove list dim
        # mv.roadgraph(data["roadgraph"], data["roadgraph_valid"])
        # mv.roadmap(data["roadmap"])
        mv.roadmap_and_occupancy(
            data["roadmap"],
            data["heatmap"],
            data["signals"],
            roi_scale=waymo.occupancy_roi,
        )
        # mv.sequence(data)
        # mv.occupancy_from_current_pose(data)
        break


if __name__ == "__main__":
    main()
