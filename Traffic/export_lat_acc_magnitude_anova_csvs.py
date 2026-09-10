#!/usr/bin/env python3
"""
Export long-format CSVs of lateral acceleration magnitude (mean |dv/dt| per segment).

Reads ``segment_level_stats.csv`` from ``style_learn_evaluate.py`` and aligns rows
across models the same way as ``plot_style_eval_results._write_lane_change_per_segment_csvs``.

Outputs (default under ``--out-dir``):
  - ``lat_acc_magnitude_per_segment_long__all.csv`` — every validation segment × model.
  - ``lat_acc_magnitude_per_segment_long__lane_change.csv`` — LCL/LCR segments only.

Columns are chosen for downstream ANOVA (task = maneuver, method = model, response =
``lat_acc_mag_mean``). Optional ``val_idx`` is joined from ``style_eval_arrays.npz``
when present (same row order as evaluation index ``j``).

Models (fixed order): original, pca_generic, pca_personal, vae_generic, vae_personal.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from Traffic.config import MNV_LCL, MNV_LCR, MNV_LK

MODELS_ORDER = (
    "original",
    "pca_generic",
    "pca_personal",
    "vae_generic",
    "vae_personal",
)

MNV_NAMES = {MNV_LK: "LK", MNV_LCL: "LCL", MNV_LCR: "LCR"}


def _load_by_model(seg_csv: Path) -> dict[str, list[dict[str, str]]]:
    df = pd.read_csv(seg_csv)
    if "model" not in df.columns or "lat_acc_mag_mean" not in df.columns:
        raise SystemExit(f"{seg_csv} must contain columns 'model', 'lat_acc_mag_mean'")
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for _, row in df.iterrows():
        by_model[str(row["model"])].append(row.to_dict())
    return by_model


def _aligned_rows(
    by_model: dict[str, list[dict[str, str]]],
) -> tuple[list[list[dict[str, str]]], int]:
    lists_mod = [by_model.get(m, []) for m in MODELS_ORDER]
    if not lists_mod[0]:
        raise SystemExit("No rows for model 'original' — cannot align.")
    n0 = len(lists_mod[0])
    if not all(len(by_model.get(m, [])) == n0 for m in MODELS_ORDER):
        raise SystemExit("Unequal row counts per model in segment_level_stats.csv")
    # rows[j][mi] = row for segment j, model MODELS_ORDER[mi]
    rows: list[list[dict[str, str]]] = []
    for j in range(n0):
        rows.append([by_model[m][j] for m in MODELS_ORDER])
    return rows, n0


def _val_idx_array(npz_path: Path | None) -> np.ndarray | None:
    if npz_path is None or not npz_path.is_file():
        return None
    z = np.load(npz_path)
    if "val_idx" not in z.files:
        return None
    return np.asarray(z["val_idx"])


def build_long_rows(
    rows_per_j: list[list[dict[str, str]]],
    val_idx: np.ndarray | None,
    lane_change_only: bool,
) -> list[dict]:
    out: list[dict] = []
    lc_counter = 0
    for j, model_rows in enumerate(rows_per_j):
        r0 = model_rows[0]
        try:
            mnv = int(float(r0.get("maneuver", -1)))
        except (TypeError, ValueError):
            continue
        if lane_change_only and mnv not in (MNV_LCL, MNV_LCR):
            continue
        seg_ix = lc_counter if lane_change_only else j
        if lane_change_only:
            lc_counter += 1
        vidx = int(val_idx[j]) if val_idx is not None and j < len(val_idx) else ""
        for row in model_rows:
            try:
                acc = float(row["lat_acc_mag_mean"])
            except (TypeError, ValueError, KeyError):
                acc = float("nan")
            out.append(
                {
                    "val_segment_index": seg_ix,
                    "val_idx": vidx,
                    "driver_id": int(float(row["driver_id"])),
                    "maneuver": mnv,
                    "maneuver_name": MNV_NAMES.get(mnv, str(mnv)),
                    "model": str(row["model"]),
                    "lat_acc_mag_mean": acc,
                }
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Export lat acc magnitude CSVs for ANOVA.")
    p.add_argument(
        "--segment-csv",
        type=str,
        default=str(_REPO / "Traffic/style_eval_results/segment_level_stats.csv"),
        help="segment_level_stats.csv from style_learn_evaluate.py",
    )
    p.add_argument(
        "--arrays",
        type=str,
        default=str(_REPO / "Traffic/style_eval_results/style_eval_arrays.npz"),
        help="Optional npz for val_idx (same segment order as j).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(_REPO / "Traffic/style_eval_results/plots"),
        help="Output directory for CSVs.",
    )
    p.add_argument(
        "--no-val-idx",
        action="store_true",
        help="Do not add val_idx column even if npz exists.",
    )
    args = p.parse_args()

    seg_path = Path(args.segment_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    npz_path = Path(args.arrays).expanduser().resolve()

    if not seg_path.is_file():
        raise SystemExit(f"Not found: {seg_path}")

    by_model = _load_by_model(seg_path)
    rows_per_j, n0 = _aligned_rows(by_model)
    val_idx = None if args.no_val_idx else _val_idx_array(npz_path)
    if val_idx is not None and len(val_idx) != n0:
        print(
            f"Warning: val_idx length {len(val_idx)} != n segments {n0}; "
            "val_idx column left blank.",
            flush=True,
        )
        val_idx = None

    out_dir.mkdir(parents=True, exist_ok=True)

    for lc_only, suffix in ((False, "__all"), (True, "__lane_change")):
        long_rows = build_long_rows(rows_per_j, val_idx, lane_change_only=lc_only)
        out_path = out_dir / f"lat_acc_magnitude_per_segment_long{suffix}.csv"
        pd.DataFrame(long_rows).to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(long_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
