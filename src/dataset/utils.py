"""
Extra utilities for calculating the maximum map size and 
the percentage of vehicles in the scene occluded
"""
from typing import Dict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from torch import Tensor
from nvidia.dali.plugin.pytorch import DALIGenericIterator


def get_cache_record_idx_path(dataset_path: Path) -> Path:
    """
    Initially try to make with tf record dali index
    in folder adjacent to dataset suffixed by idx.
    If that fails due to permission requirements, make in /tmp.
    """
    dali_idx_path = dataset_path.parent / f"{dataset_path.name}_dali_idx"
    if not dali_idx_path.exists():
        try:
            dali_idx_path.mkdir()
            return dali_idx_path
        except OSError:
            print(
                f"Unable to create dali index at {dali_idx_path},"
                f" changing to /tmp/{dataset_path.name}_dali_idx"
            )

            dali_idx_path = Path(f"/tmp/{dataset_path.name}_dali_idx")
            if not dali_idx_path.exists():
                dali_idx_path.mkdir()

    return dali_idx_path


def check_maximum_map_size(dataloaders: Dict[str, DALIGenericIterator]) -> None:
    """
    Determine the maximum map size for waymo motion.
    This is required for calibration of the output map.
    Such that it can cover the entire range of data.
    """

    max_x_range, max_y_range = torch.tensor(0).cuda(), torch.tensor(0).cuda()
    with tqdm(
        total=sum(len(d) for d in dataloaders.values()), unit="batch", ncols=200
    ) as pbar:
        for dataloader in dataloaders.values():
            for data in dataloader:
                data: Dict[str, Tensor] = data[0]
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
                max_x_range = torch.max((x_max - x_min).max(), max_x_range)
                max_y_range = torch.max((y_max - y_min).max(), max_y_range)
                pbar.update(1)
                pbar.set_description(
                    f"max map size {max_x_range:.1f}x{max_y_range:.1f}", refresh=True
                )


def evaluate_vehicle_occulsions(
    dataloaders: Dict[str, DALIGenericIterator], max_samples: int = 0
) -> None:
    """Determine the percentage of observed vehciles remaining at later timesteps"""

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

                valid_agents: Tensor = data[0]["agents_valid"]  # B,N,T

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
    # Plot Output and save figure
    from matplotlib import pyplot as plt

    plt.figure(figsize=(6, 3))
    plt.title("Observed Agent Decay")
    plt.plot(np.arange(0, len(waypoints)), (waypoint_decay * 100).cpu().numpy(), lw=5)
    plt.ylim((40, 100))
    plt.xlabel("Time after present (sec)")
    plt.ylabel("% Agents Observed")
    plt.tight_layout()
    plt.savefig("waymo_agent_decay.png")


def velocity_distribution(loader: DALIGenericIterator, n_samples: int) -> None:
    """Sampe the data a few times and find an
    approximate mean-variance to vehicle velocity"""
    all_data = []
    for idx, data in tqdm(enumerate(loader), total=n_samples // loader.batch_size + 1):
        data: Dict[str, Tensor] = data[0]
        vel = data["agents"][data["agents_valid"] > 0][:, 3:5]
        all_data.append(vel)

        if idx * loader.batch_size > n_samples:
            break

    all_data = torch.cat(all_data)
    print(f"mean {all_data.mean(dim=0)}, var: {all_data.var(dim=0)}")
