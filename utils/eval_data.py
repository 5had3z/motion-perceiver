from dataclasses import dataclass, field
from typing import List
import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np
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
        return f"{self.name} - IoU: {self.iou:.3f}, AUC: {self.auc:.3f}"


def get_db(path: Path):
    """Return handle to db, adds base tables if it didn't previously exist"""
    create_table = not path.exists()
    con = sqlite3.connect(path)

    if create_table:
        con.execute(
            "CREATE TABLE metadata (hash TEXT PRIMARY KEY, ts TIMESTAMP, desc TEXT, epoch INT)"
        )
        for table in ["pytorch", "tensorflow"]:
            con.execute(
                f"CREATE TABLE {table} (hash TEXT PRIMARY KEY, epoch INT, iou FLOAT, auc FLOAT)",
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
    con = sqlite3.connect(workspace / "waymo_eval.db")
    meta = pd.read_sql_query("SELECT epoch, hash FROM metadata", con, index_col="hash")
    perf = pd.read_sql_query("SELECT epoch, hash FROM pytorch", con, index_col="hash")
    con.close()

    missing = meta.index.difference(perf.index).to_list()
    outdated = meta.gt(perf).query("epoch").index.to_list()

    return missing + outdated
