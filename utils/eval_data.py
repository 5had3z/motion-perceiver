import itertools
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Iterable

import numpy as np
import pandas as pd
from konductor.utilities.metadata import Metadata


@dataclass
class MetricData:
    name: str
    _iou: List[float] = field(default_factory=list)
    _auc: List[float] = field(default_factory=list)

    def add_iou(self, iou: float) -> None:
        self._iou.append(iou)

    def add_auc(self, auc: float) -> None:
        self._auc.append(auc)

    @property
    def auc(self) -> float:
        return np.array(self._auc).mean()

    @property
    def iou(self) -> float:
        return np.array(self._iou).mean()

    def __str__(self) -> str:
        return f"{self.name} IoU: {self.iou:.3f}, AUC: {self.auc:.3f}"


def metric_data_list_to_dict(metric_data: Iterable[MetricData]) -> Dict[str, float]:
    """Reformat the list of metric data to a dict"""
    ret: Dict[str, float] = {}
    for sample in metric_data:
        suffix = sample.name.split("_")[1]
        ret[f"auc_{suffix}"] = sample.auc
        ret[f"iou_{suffix}"] = sample.iou
    return ret


def create_waypoints_table(cur: sqlite3.Cursor):
    """Create table for waypoint performance if it doesn't already exist"""
    table_cmd = "CREATE TABLE IF NOT EXISTS waypoints "
    table_cmd += "(hash TEXT PRIMARY KEY, epoch INT, iteration INT"

    # Create list of timestamps + mean (for convenience)
    for key, ts in itertools.product(["iou", "auc"], list(range(1, 9)) + ["mean"]):
        table_cmd += f", {key}_{ts} FLOAT"
    table_cmd += ")"

    cur.execute(table_cmd)


def upsert_waypoints(
    cur: sqlite3.Cursor, run_hash: str, metadata: Metadata, data: Dict[str, float]
):
    """Upsert experiment waypoints performance"""

    # Split into key value lists
    keys: List[str] = []
    values: List[float] = []
    for k, v in data.items():
        keys.append(k)
        values.append(v)

    # Create strings for sql command
    key_string = ", ".join(keys)
    set_string = ", ".join([f"{k}=excluded.{k}" for k in keys])

    cur.execute(
        f"INSERT INTO waypoints (hash, epoch, iteration, {key_string}) "
        f"VALUES (?, ?, ?{', ?' * len(keys)}) ON CONFLICT (hash) DO UPDATE "
        f"SET epoch=excluded.epoch, iteration=excluded.iteration, {set_string};",
        [run_hash, metadata.epoch, metadata.iteration, *values],
    )


def get_db(path: Path):
    """Return handle to db, adds base tables if it didn't previously exist"""
    con = sqlite3.connect(path / "results.db")

    con.execute(
        "CREATE TABLE IF NOT EXISTS metadata (hash TEXT PRIMARY KEY, ts TIMESTAMP, desc TEXT, epoch INT)"
    )
    for table in ["pytorch", "tensorflow"]:
        con.execute(
            f"CREATE TABLE IF NOT EXISTS {table} (hash TEXT PRIMARY KEY, epoch INT, iou FLOAT, auc FLOAT)",
        )
    con.commit()

    return con


def upsert_metadata(cur: sqlite3.Cursor, run_hash: str, metadata: Metadata):
    """Upsert operation on experiment metadata table"""
    cur.execute(
        f"INSERT INTO metadata (hash, ts, desc, epoch) "
        "VALUES (?, ?, ?, ?) ON CONFLICT (hash) DO UPDATE "
        "SET ts=excluded.ts, desc=excluded.desc, epoch=excluded.epoch;",
        [run_hash, metadata.train_last, metadata.brief, metadata.epoch],
    )


def upsert_eval(cur: sqlite3.Cursor, run_hash: str, epoch: int, data: MetricData):
    """Upsert experiment iou/auc performance"""
    cur.execute(
        f"INSERT INTO {data.name} (hash, epoch, iou, auc) "
        "VALUES (?, ?, ?, ?) ON CONFLICT (hash) DO UPDATE "
        "SET epoch=excluded.epoch, iou=excluded.iou, auc=excluded.auc;",
        [run_hash, epoch, data.iou, data.auc],
    )


def find_outdated_runs(workspace: Path) -> List[str]:
    con = sqlite3.connect(workspace / "results.db")
    meta = pd.read_sql_query("SELECT epoch, hash FROM metadata", con, index_col="hash")
    perf = pd.read_sql_query("SELECT epoch, hash FROM pytorch", con, index_col="hash")
    con.close()

    missing = meta.index.difference(perf.index).to_list()
    outdated = meta.gt(perf).query("epoch").index.to_list()

    return missing + outdated
