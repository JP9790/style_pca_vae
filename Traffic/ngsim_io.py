"""Load and filter NGSIM CSV (chunked) for selected ``Location`` values."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from Traffic.config import DEFAULT_LOCATIONS

REQUIRED_COLS = (
    "Vehicle_ID",
    "Frame_ID",
    "Global_Time",
    "Local_X",
    "Local_Y",
    "Lane_ID",
    "Location",
)


def load_ngsim_subset(
    csv_path: str | Path,
    locations: Iterable[str] | None = None,
    chunksize: int = 400_000,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    Read ``Next_Generation_Simulation_*`` vehicle trajectories, keeping only
    ``Location`` in ``locations`` (case-insensitive), e.g. ``us-101``, ``i-80``.

    If ``max_rows`` is set, stop after accumulating that many rows **after**
    the location filter (not raw CSV rows), so early file regions without
    target locations still scan forward until enough matching rows are found.
    """
    csv_path = Path(csv_path)
    locs = tuple(x.lower().strip() for x in (locations or DEFAULT_LOCATIONS))
    chunks: list[pd.DataFrame] = []
    total_filtered = 0
    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunksize,
        usecols=list(REQUIRED_COLS),
        dtype={"Vehicle_ID": np.int64, "Global_Time": np.int64},
        thousands=",",
        low_memory=False,
    ):
        loc = chunk["Location"].astype(str).str.lower().str.strip()
        sub = chunk[loc.isin(locs)].copy()
        if len(sub) and max_rows is not None:
            room = max_rows - total_filtered
            if room <= 0:
                break
            if len(sub) > room:
                sub = sub.iloc[:room].copy()
        if len(sub):
            chunks.append(sub)
            total_filtered += len(sub)
        if max_rows is not None and total_filtered >= max_rows:
            break
    if not chunks:
        raise SystemExit(f"No rows with Location in {locs} under {csv_path}")
    out = pd.concat(chunks, ignore_index=True)
    out.sort_values(["Vehicle_ID", "Global_Time"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def train_val_split_indices(
    n: int, val_fraction: float = 0.1, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_fraction)))
    return idx[n_val:], idx[:n_val]
