"""
Train / validation split for NGSIM segment datasets (``prepare_dataset.py`` output).

Implements a **driver-grouped, maneuver-level holdout** aligned with the described
evaluation protocol:

1. Segments are grouped by driver (``vehicle_id``).
2. For each driver, **one** maneuver label among {lane keeping, lane change left,
   lane change right} is chosen to be **fully absent** from training. All segments of
   that maneuver go to validation so style learning never sees that maneuver for that
   driver.
3. To target **approximately** ``holdout_fraction`` of each driver's segments in
   validation **while** keeping evaluation **diverse**, any shortfall vs. that target
   is filled by **stratified** random sampling from the remaining (train-eligible)
   segments (so validation can include multiple maneuver types when needed).
4. If the excluded maneuver already accounts for **more** than the target fraction,
   validation is larger than the target (the unseen-maneuver constraint takes
   precedence).

Outputs global row indices into ``X`` / ``y`` / ``vehicle_id`` from the input ``.npz``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from Traffic.config import MNV_LCL, MNV_LCR, MNV_LK

ManeuverNames = {MNV_LK: "LK", MNV_LCL: "LCL", MNV_LCR: "LCR"}


def _stratified_sample(
    indices: np.ndarray,
    y_local: np.ndarray,
    n_take: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``n_take`` indices from ``indices`` with proportional allocation per label."""
    if n_take <= 0 or len(indices) == 0:
        return np.array([], dtype=np.int64)
    if n_take >= len(indices):
        return indices.astype(np.int64, copy=False)

    labs = np.unique(y_local)
    counts = np.array([(y_local == lab).sum() for lab in labs], dtype=np.float64)
    total = float(counts.sum())
    raw = n_take * counts / total
    quotas = np.floor(raw).astype(int)
    rem = int(n_take - quotas.sum())
    frac = raw - quotas
    order = np.argsort(-frac)
    for j in range(rem):
        quotas[order[j % len(quotas)]] += 1

    parts: list[np.ndarray] = []
    for lab, q in zip(labs, quotas):
        if q <= 0:
            continue
        lab_idx = indices[y_local == lab]
        q = min(int(q), len(lab_idx))
        parts.append(rng.choice(lab_idx, size=q, replace=False))
    out = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
    if len(out) < n_take:
        # Rare rounding shortfall: fill without replacement from remainder
        pool = np.setdiff1d(indices, out, assume_unique=False)
        need = n_take - len(out)
        if len(pool) > 0 and need > 0:
            extra = rng.choice(pool, size=min(need, len(pool)), replace=False)
            out = np.concatenate([out, extra])
    return out[:n_take]


def _choose_excluded_maneuver(
    y_local: np.ndarray,
    rng: np.random.Generator,
    strategy: Literal["closest_to_fraction", "random"] = "closest_to_fraction",
    holdout_fraction: float = 0.3,
) -> int:
    """Pick one maneuver type to hold out entirely from training for this driver."""
    labs = np.unique(y_local)
    if len(labs) < 2:
        raise ValueError("Driver needs at least 2 maneuver types for this split.")

    if strategy == "random":
        return int(rng.choice(labs))

    n_total = len(y_local)
    best: list[int] = []
    best_err = 1e9
    for lab in labs:
        n_m = int((y_local == lab).sum())
        err = abs(n_m / max(n_total, 1) - holdout_fraction)
        if err < best_err - 1e-9:
            best_err = err
            best = [int(lab)]
        elif abs(err - best_err) <= 1e-9:
            best.append(int(lab))
    return int(rng.choice(np.array(best, dtype=np.int64)))


def maneuver_holdout_split(
    vehicle_id: np.ndarray,
    y: np.ndarray,
    holdout_fraction: float = 0.3,
    seed: int = 0,
    excluded_strategy: Literal["closest_to_fraction", "random"] = "closest_to_fraction",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Build global train/val index arrays.

    Parameters
    ----------
    vehicle_id, y
        Same length ``N`` as in ``prepare_dataset`` output.
    holdout_fraction
        Target fraction of each driver's segments in validation (excluding maneuver
        constraint; excluded maneuver may push validation above this).
    seed
        RNG seed for reproducibility.
    excluded_strategy
        How to pick the fully held-out maneuver type per driver.

    Returns
    -------
    train_idx, val_idx
        Disjoint 1D int64 indices covering ``0..N-1``.
    meta
        One dict per driver with counts and excluded label.
    """
    vehicle_id = np.asarray(vehicle_id).ravel()
    y = np.asarray(y).ravel()
    if vehicle_id.shape != y.shape:
        raise ValueError("vehicle_id and y must have the same shape.")
    n = len(y)
    drivers = np.unique(vehicle_id)

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []

    for vid in drivers:
        rng = np.random.default_rng((seed * 1_000_003 + int(vid)) % (2**32))
        idx = np.flatnonzero(vehicle_id == vid)
        y_loc = y[idx]
        n_d = len(idx)

        excluded = _choose_excluded_maneuver(
            y_loc, rng, strategy=excluded_strategy, holdout_fraction=holdout_fraction
        )

        unseen_mask = y_loc == excluded
        unseen = idx[unseen_mask]
        pool = idx[~unseen_mask]

        n_val_target = int(round(holdout_fraction * n_d))
        k_extra = max(0, n_val_target - len(unseen))
        extra = _stratified_sample(pool, y_loc[~unseen_mask], k_extra, rng)

        val_local = np.concatenate([unseen, extra]) if len(extra) else unseen
        val_set = set(int(i) for i in val_local.tolist())
        train_local = np.array([i for i in idx.tolist() if i not in val_set], dtype=np.int64)

        val_parts.append(val_local.astype(np.int64, copy=False))
        train_parts.append(train_local)

        meta.append(
            {
                "vehicle_id": int(vid),
                "excluded_maneuver": int(excluded),
                "excluded_name": ManeuverNames.get(int(excluded), str(excluded)),
                "n_segments": int(n_d),
                "n_val_unseen_maneuver": int(len(unseen)),
                "n_val_supplement": int(len(extra)),
                "n_val": int(len(val_local)),
                "n_train": int(len(train_local)),
                "val_fraction": float(len(val_local) / max(n_d, 1)),
            }
        )

    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)

    train_idx.sort()
    val_idx.sort()

    return train_idx, val_idx, meta


def verify_split(
    vehicle_id: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    meta: list[dict[str, Any]],
) -> None:
    """Assert disjoint full cover and zero train samples on excluded maneuver per driver."""
    n = len(vehicle_id)
    tr = set(int(i) for i in np.asarray(train_idx).tolist())
    va = set(int(i) for i in np.asarray(val_idx).tolist())
    assert tr.isdisjoint(va), "train/val overlap"
    assert tr | va == set(range(n)), "train/val must partition all indices"

    excluded_map = {m["vehicle_id"]: m["excluded_maneuver"] for m in meta}
    for i in train_idx:
        vid = int(vehicle_id[i])
        ex = excluded_map[vid]
        assert int(y[i]) != ex, f"train contains excluded maneuver {ex} for vehicle {vid}"


def load_segments_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    return data["vehicle_id"], data["y"], data["X"]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Maneuver-level holdout train/val split (by driver)."
    )
    p.add_argument(
        "--input",
        type=str,
        default=str(_REPO / "Traffic/ngsim_us101_i80_segments.npz"),
        help="Path to prepare_dataset .npz (must contain vehicle_id, y, X).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(_REPO / "Traffic/train_val_split.npz"),
        help="Output .npz with train_idx, val_idx and metadata.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.3,
        help="Target validation fraction per driver (before unseen-maneuver constraint).",
    )
    p.add_argument(
        "--excluded-strategy",
        choices=("closest_to_fraction", "random"),
        default="closest_to_fraction",
        help="How to pick the fully held-out maneuver type per driver.",
    )
    p.add_argument(
        "--meta-json",
        type=str,
        default=None,
        help="Optional path to write per-driver metadata as JSON.",
    )
    p.add_argument(
        "--embed-data",
        action="store_true",
        help="Also store X, y, vehicle_id in the output (duplicates the input file).",
    )
    args = p.parse_args()

    inp = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    vehicle_id, y, X = load_segments_npz(inp)
    train_idx, val_idx, meta = maneuver_holdout_split(
        vehicle_id,
        y,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        excluded_strategy=args.excluded_strategy,
    )
    verify_split(vehicle_id, y, train_idx, val_idx, meta)

    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "holdout_fraction": np.float32(args.holdout_fraction),
        "seed": np.int32(args.seed),
        "excluded_strategy": np.array(args.excluded_strategy),
        "meta_json": np.array(json.dumps(meta), dtype=object),
        "source_npz": np.array(str(inp), dtype=object),
    }
    if args.embed_data:
        payload["vehicle_id"] = vehicle_id
        payload["y"] = y
        payload["X"] = X
    np.savez_compressed(out, **payload)
    print(
        f"Saved {out}\n"
        f"  N={len(y)}  train={len(train_idx)}  val={len(val_idx)}\n"
        f"  drivers={len(meta)}  holdout_fraction={args.holdout_fraction}",
        flush=True,
    )
    if args.meta_json:
        mj = Path(args.meta_json).expanduser().resolve()
        mj.parent.mkdir(parents=True, exist_ok=True)
        mj.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  metadata JSON: {mj}", flush=True)


if __name__ == "__main__":
    main()
