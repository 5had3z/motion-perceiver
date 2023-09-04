from pathlib import Path
from typing import List

import pandas as pd
from konductor.webserver.utils import fill_experiments, Experiment


def data_by_time(exp: Experiment, split: str, key: str) -> pd.Series:
    """Iterate over dataset and transform to be indexed by time"""

    # Get latest iteration performance data
    data = exp.get_group_latest(split, "occupancy")

    # Filter to f"key_{timestep}"
    data = data.filter(items=[s for s in data.columns if key in s])

    # Transform to be indexed by timestep with "key" as label name
    data.rename(
        {s: int(s.split("_")[-1]) for s in data.columns}, axis="columns", inplace=True
    )
    data = data.transpose().sort_index()

    series: pd.Series = data[data.columns.values[0]]
    series.name = exp.name

    return series


def gather_experiment_time_performance(
    exps: List[Experiment], split: str, metric: str
) -> List[pd.Series]:
    """For each experiment with the timeseries metric, gather the data from the last epoch and transform into a time series to plot prediction performance over time"""

    data: List[pd.Series] = [
        data_by_time(e, split, metric)
        for e in exps
        if any(metric in s for s in e.stats)  # if the metric appears in exp keys
    ]

    return data


def debug() -> None:
    """dash is annoying to debug, lets separate debugging of functions that gather data"""
    experiments = []
    fill_experiments(Path("/media/bryce/2TB_Seagate/motion-perceiver"), experiments)
    exp_perf = gather_experiment_time_performance(experiments, "val", "IoU")
    print(exp_perf)


if __name__ == "__main__":
    debug()
