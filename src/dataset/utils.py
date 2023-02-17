from pathlib import Path
from typing import Any, Dict
import os

from easydict import EasyDict
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import torch


from nnet_training.utilities import comm

try:
    from .waymo import waymo_motion_pipe
    from .interaction import interation_pipeline
    from . import visualisation as mv
except ImportError:
    from waymo import waymo_motion_pipe
    from interaction import interation_pipeline
    import visualisation as mv


def get_dali_dataloader(
    dataset_config: EasyDict, num_threads: int, val_only: bool = False
) -> Dict[str, DALIGenericIterator]:
    """Return DALIGenericIterator for as Dataloader, mostly used for tf.Record datasets"""
    pipe_kwargs = {
        "shard_id": comm.get_rank(),
        "num_shards": comm.get_world_size(),
        "num_threads": num_threads,
        "device_id": torch.cuda.current_device(),
        "batch_size": dataset_config.batch_size // comm.get_world_size(),
    }

    dataset_name = dataset_config.type.lower()
    if dataset_name == "waymo_motion":
        waymo_root = Path(os.environ.get("DATA_PATH", "/data"))
        pipes = [
            waymo_motion_pipe(
                record_root=waymo_root / "validation",
                random_shuffle=False,
                **pipe_kwargs,
                **dataset_config.args,
            )
        ]
        reader_names = ["waymo_validation"]
        if not val_only:
            pipes.append(
                waymo_motion_pipe(
                    record_root=waymo_root / "training",
                    random_shuffle=dataset_config.shuffle,
                    **pipe_kwargs,
                    **dataset_config.args,
                )
            )
            reader_names.append("waymo_training")

        output_map = ["agents", "agents_valid"]
        if dataset_config.args.get("road_features", False):
            output_map.extend(["roadgraph", "roadgraph_valid"])
        if dataset_config.args.get("roadmap", False):
            output_map.append("roadmap")
        if dataset_config.args.get("signal_features", False):
            output_map.extend(["signals", "signals_valid"])
        if dataset_config.args.get("occupancy_size", 0) > 0:
            output_map.extend(["heatmap", "time_idx"])
        if dataset_config.args.get("scenario_id", False):
            output_map.append("scenario_id")

    elif dataset_name == "interaction":
        interaction_root = (
            Path(os.environ.get("INTERACTION_PATH", "/data")) / "tfrecord"
        )
        pipes = [
            interation_pipeline(
                record_file=interaction_root / "interaction_val.tfrecord",
                random_shuffle=False,
                **pipe_kwargs,
                **dataset_config.args,
            )
        ]
        reader_names = ["interaction_val"]
        if not val_only:
            pipes.append(
                interation_pipeline(
                    record_file=interaction_root / "interaction_train.tfrecord",
                    random_shuffle=dataset_config.shuffle,
                    **pipe_kwargs,
                    **dataset_config.args,
                )
            )
            reader_names.append("interaction_train")

        output_map = ["agents", "agents_valid"]
        if dataset_config.args.get("roadmap", False):
            output_map.append("roadmap")
        if dataset_config.args.get("occupancy_size", 0) > 0:
            output_map.extend(["heatmap", "time_idx"])
    else:
        raise NotImplementedError(dataset_name)

    dataloaders = {}
    for pipe, rname, name in zip(pipes, reader_names, ["Validation", "Training"]):
        pipe.build()
        dataloaders[name] = DALIGenericIterator(
            pipe, output_map, reader_name=rname, auto_reset=True
        )

    return dataloaders


def check_maximum_map_size(dataloaders) -> None:
    """
    Determine the maximum map size for waymo motion.
    This is required for calibration of the output map.
    Such that it can cover the entire range of data.
    """
    from tqdm.auto import tqdm

    max_x_range, max_y_range = torch.tensor(0).cuda(), torch.tensor(0).cuda()
    with tqdm(
        total=sum(len(d) for d in dataloaders.values()), unit="batch", ncols=200
    ) as pbar:
        for dataloader in dataloaders.values():
            for data in dataloader:
                data: Dict[str, torch.Tensor] = data[0]
                alldata = torch.cat(
                    [data["past"], data["current"], data["future"]], dim=2
                )
                mask = alldata[..., 0:1].bool()
                xy_data = alldata[..., 1:3]

                # get batch-wise max values
                filt = torch.where(
                    mask.expand_as(xy_data),
                    xy_data,
                    torch.tensor(-torch.inf).cuda().expand_as(xy_data),
                )
                x_max = filt[..., 0].max(dim=-1)[0]
                y_max = filt[..., 1].max(dim=-1)[0]

                # get batch-wise min values
                filt = torch.where(
                    mask.expand_as(xy_data),
                    xy_data,
                    torch.tensor(torch.inf).cuda().expand_as(xy_data),
                )
                x_min = filt[..., 0].min(dim=-1)[0]
                y_min = filt[..., 1].min(dim=-1)[0]

                # Find range
                max_x_range = max((x_max - x_min).max(), max_x_range)
                max_y_range = max((y_max - y_min).max(), max_y_range)
                pbar.update(1)
                pbar.set_description(
                    f"max map size {max_x_range:.1f}x{max_y_range:.1f}", refresh=True
                )


def evaluate_vehicle_occulsions(
    dataloaders: Dict[str, Any], max_samples: int = 0
) -> None:
    """Determine the percentage of observed vehciles remaining at later timesteps"""
    from tqdm.auto import tqdm
    import numpy as np
    from matplotlib import pyplot as plt

    n_samples = (
        10000
        if max_samples < 1
        else sum(len(d) * d.batch_size for d in dataloaders.values())
    )
    waypoints = list(range(10, 91, 10))
    waypoint_decay = torch.zeros((len(waypoints))).cuda()

    with tqdm(total=n_samples, unit="batch", ncols=100) as pbar:
        for dataloader in dataloaders.values():
            for data in dataloader:
                if pbar.n >= n_samples:
                    n_samples = pbar.n  # change to actual samples
                    break  # jump out at max samples

                valid_agents: torch.Tensor = data[0]["agents_valid"]  # B,N,T

                # mask out occluded agents (not observed in past or present)
                observed_agents = valid_agents[..., 0:11].sum(dim=2) > 0
                valid_agents[~observed_agents] = 0

                # count unoccluded agents in scene
                # 1. sum over time dim
                # 2. if greater than zero is agent
                # 3. sum bools over agents dim to get n agents
                n_total = (valid_agents.sum(dim=2) > 0).sum(dim=1, keepdim=True)

                # count agents at time point
                # 1. sum bools over agent dim to get agents at time
                n_waypoint = valid_agents[..., waypoints].sum(dim=1)

                # decay is number of agents at time point div by total in scene
                # sum over batch dim to accumulate
                waypoint_decay += (n_waypoint / n_total).sum(dim=0)

                pbar.update(valid_agents.shape[0])

    waypoint_decay /= n_samples
    print(waypoint_decay)
    plt.figure(figsize=(6, 3))
    plt.title("Observed Agent Decay")
    plt.plot(np.arange(0, len(waypoints)), (waypoint_decay * 100).cpu().numpy(), lw=5)
    plt.ylim((40, 100))
    plt.xlabel("Time after present (sec)")
    plt.ylabel("% Agents Observed")
    plt.tight_layout()
    plt.savefig("waymo_agent_decay.png")


def test_dali_loader() -> None:
    """Yield some data"""

    loader_config = {
        "type": "waymo_motion",
        "batch_size": 4,
        "shuffle": True,
        "drop_last": True,
        "n_workers": 8,
        "args": {
            "map_normalize": 40.0,
            "full_sequence": True,
            "occupancy_size": 256,
            "heatmap_time": list(range(0, 91, 10)),
            "filter_future": True,
            # "separate_classes": True,
            # "random_heatmap_count": 4,
            # "random_heatmap_minmax": [1, 40],
            "signal_features": True,
            "roadmap": True,
            "use_sdc_frame": True,
            "waymo_eval_frame": True,
        },
    }

    loaders = get_dali_dataloader(loader_config, num_threads=loader_config["n_workers"])

    # evaluate_vehicle_occulsions(loaders, 0)
    # return

    # check_maximum_map_size(loaders)
    # return

    for data in loaders["Validation"]:
        data: Dict[str, torch.Tensor] = data[0]  # remove list dim
        # visualise_roadgraph(data["roadgraph"], data["roadgraph_valid"])
        # mv.visualise_roadmap(data["roadmap"])
        # mv.overlay_roadmap_and_occupancy(
        #     data["roadmap"], data["heatmap"], data["signals"]
        # )
        mv.visualize_sequence(data)
        # visualise_occupancy_map(data)
        break


if __name__ == "__main__":
    test_dali_loader()
