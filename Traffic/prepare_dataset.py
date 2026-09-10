#!/usr/bin/env python3
"""
Prepare NGSIM US-101 / I-80 trajectory segments for style / imitation learning.

Pipeline (see module docstrings in ``Traffic/``):

1. Load CSV, filter ``Location`` to ``us-101`` and ``i-80``.
2. Per vehicle: sort by time; drop tracks shorter than 8 s.
3. Bicycle EKF smoothing on ``(Local_X, Local_Y)``; recompute velocity / acceleration.
4. Segment maneuvers: lane change (±3 s around lane-id change); lane keeping via sliding 6 s
   windows along stable-lane runs (stride in ``Traffic.config.LK_STRIDE_STEPS``) with heading check.
5. Retain drivers with ≥10 segments and ≥2 maneuver types among {LK, LCL, LCR}.
6. Task-aligned Frenet-style frame: lateral displacement and lateral velocity profiles,
   resampled to 60 points at 10 Hz; optional z-score per segment.

Example::

    python Traffic/prepare_dataset.py \\
        --csv Next_Generation_Simulation_\\(NGSIM\\)_Vehicle_Trajectories_and_Supporting_Data_20251027.csv \\
        --output Traffic/ngsim_us101_i80_segments.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Repo root on sys.path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from Traffic.config import (
    DT_S,
    MIN_MANEUVER_TYPES,
    MIN_SEGMENTS_PER_DRIVER,
    MIN_TRACK_STEPS,
    MNV_LCL,
    MNV_LCR,
    MNV_LK,
    N_RESAMPLE,
)
from Traffic.ekf_smooth import smooth_track_bicycle_ekf
from Traffic.frenet import task_aligned_frenet_profiles, zscore_per_segment
from Traffic.ngsim_io import load_ngsim_subset
from Traffic.segmentation import (
    extract_lane_change_segments,
    extract_lane_keeping_segments,
    SegmentIndex,
)


def _vehicle_passes_filters(
    maneuver_list: list[int],
) -> bool:
    if len(maneuver_list) < MIN_SEGMENTS_PER_DRIVER:
        return False
    kinds = set(maneuver_list)
    if len(kinds) < MIN_MANEUVER_TYPES:
        return False
    return True


def _append_segment_features(
    out_X: list,
    out_y: list,
    out_vid: list,
    xy_seg: np.ndarray,
    maneuver: int,
    vehicle_id: int,
    zscore: bool,
) -> None:
    d, v_d = task_aligned_frenet_profiles(xy_seg, DT_S, N_RESAMPLE)
    if zscore:
        d = zscore_per_segment(d)
        v_d = zscore_per_segment(v_d)
    feat = np.stack([d, v_d], axis=-1)
    out_X.append(feat.astype(np.float32))
    out_y.append(int(maneuver))
    out_vid.append(int(vehicle_id))


def run(
    csv_path: Path,
    output_npz: Path,
    zscore: bool,
    max_vehicles: int | None,
    max_rows: int | None,
) -> None:
    print("Loading & filtering CSV (US-101, I-80)...", flush=True)
    df = load_ngsim_subset(csv_path, max_rows=max_rows)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    vid_list: list[int] = []

    grouped = df.groupby("Vehicle_ID", sort=False)
    n_groups = grouped.ngroups
    processed = 0

    for vehicle_id, g in grouped:
        processed += 1
        if max_vehicles is not None and processed > max_vehicles:
            break
        if len(g) < MIN_TRACK_STEPS:
            continue

        g = g.sort_values("Global_Time")
        xy = g[["Local_X", "Local_Y"]].to_numpy(dtype=np.float64)
        lane_ids = g["Lane_ID"].to_numpy(dtype=np.int64)

        xy_f, _, _, _ = smooth_track_bicycle_ekf(xy, DT_S)

        seg_lc = extract_lane_change_segments(lane_ids)
        seg_lk = extract_lane_keeping_segments(lane_ids, xy_f)
        segs: list[tuple[SegmentIndex, str]] = [(s, "lc") for s in seg_lc] + [
            (s, "lk") for s in seg_lk
        ]

        maneuvers: list[int] = []
        for s, _ in segs:
            maneuvers.append(s.maneuver)

        if not _vehicle_passes_filters(maneuvers):
            continue

        for s, _ in segs:
            sub = xy_f[s.i0 : s.i1 + 1]
            if sub.shape[0] != N_RESAMPLE:
                continue
            _append_segment_features(X_list, y_list, vid_list, sub, s.maneuver, int(vehicle_id), zscore)

        if processed % 500 == 0:
            print(f"  vehicles scanned: {processed}/{n_groups}, segments: {len(X_list)}", flush=True)

    if not X_list:
        raise SystemExit("No segments collected. Relax filters or check data paths.")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int8)
    vehicle_ids = np.array(vid_list, dtype=np.int64)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        X=X,
        y=y,
        vehicle_id=vehicle_ids,
        feature_channels=np.array(["lateral_d", "lateral_v"], dtype=object),
        dt=DT_S,
        n_resample=N_RESAMPLE,
    )
    print(
        f"Saved {output_npz}  X={X.shape}  y unique={np.unique(y)}  drivers={np.unique(vehicle_ids).size}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="NGSIM US-101 / I-80 segment dataset.")
    p.add_argument(
        "--csv",
        type=str,
        default=str(_REPO / "Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20251027.csv"),
        help="Path to NGSIM vehicle trajectory CSV.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(_REPO / "Traffic/ngsim_us101_i80_segments.npz"),
        help="Output .npz path.",
    )
    p.add_argument(
        "--zscore",
        action="store_true",
        help="Z-score lateral d and v_d within each segment.",
    )
    p.add_argument(
        "--max_vehicles",
        type=int,
        default=None,
        help="Optional cap on number of vehicle groups to scan (debug).",
    )
    p.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional cap on rows kept after US-101/I-80 filter (debug).",
    )
    args = p.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    run(
        csv_path,
        out_path,
        zscore=args.zscore,
        max_vehicles=args.max_vehicles,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
