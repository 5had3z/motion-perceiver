from dataclasses import dataclass, field
from typing import List
import sqlite3
from pathlib import Path

import numpy as np


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
    create_table = not path.exists()
    con = sqlite3.connect(path)

    if create_table:
        for table in ["pytorch", "tensorflow"]:
            con.execute(
                f"CREATE TABLE {table} (hash TEXT PRIMARY KEY, epoch INT, iou REAL, auc REAL)",
            )
        con.commit()

    return con


def update_eval_db(cur: sqlite3.Cursor, run_hash: str, epoch: int, data: MetricData):
    """Updates existing entry for experiment or creates new one if doesn't exist i.e. UPSERT"""
    cur.execute(
        f"INSERT INTO {data.name} (hash, epoch, iou, auc) "
        "VALUES (?, ?, ?, ?) ON CONFLICT (hash) DO UPDATE "
        "SET epoch=excluded.epoch, iou=excluded.iou, auc=excluded.auc;",
        [run_hash, epoch, data.iou, data.auc],
    )
