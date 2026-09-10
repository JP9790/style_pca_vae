#!/usr/bin/env python3
"""
DTW alignment RMSE between held-out trajectories and learned trajectories.

For each validation segment, multivariate DTW is run on (lateral d, lateral v) with
squared Euclidean step cost ||y_t - y_hat_t'||^2. The optimal warping path π of
length K is backtracked; then:

  RMSE_DTW,d = sqrt( (1/K) sum_k (d_tk - d_hat_t'k)^2 )
  RMSE_DTW,v = sqrt( (1/K) sum_k (v_tk - v_hat_t'k)^2 )

using the *same* path for both (path from joint (d,v) DTW).

Outputs (CSV):
  - Aggregate: mean, sample std, and standard error of the mean (std / sqrt(n)) over
    segments per method (Generic PCA, Personalized-PCA, Personalized-VAE vs ground truth).
  - Long: one row per segment × method for ANOVA-style analysis.

Reads ``style_eval_arrays.npz`` from ``style_learn_evaluate.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from Traffic.config import MNV_LCL, MNV_LCR, MNV_LK


def _dtw_path_sqeuclidean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Multivariate DTW with cost C[i,j] = ||x[i] - y[j]||^2.

    x, y: (T, C) same T.
    Returns path as (K, 2) int array of (i, j) pairs from (0,0) to (T-1,T-1).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    t = x.shape[0]
    # Local squared Euclidean
    # cost[i,j] = sum_c (x[i,c]-y[j,c])^2
    diff = x[:, None, :] - y[None, :, :]
    cost = np.sum(diff * diff, axis=2)

    inf = np.inf
    dp = np.full((t, t), inf, dtype=np.float64)
    dp[0, 0] = cost[0, 0]
    for i in range(1, t):
        dp[i, 0] = cost[i, 0] + dp[i - 1, 0]
    for j in range(1, t):
        dp[0, j] = cost[0, j] + dp[0, j - 1]
    for i in range(1, t):
        for j in range(1, t):
            dp[i, j] = cost[i, j] + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    # Backtrack (i,j) -> (0,0)
    path_rev: list[tuple[int, int]] = []
    i, j = t - 1, t - 1
    while i > 0 or j > 0:
        path_rev.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            opts = [
                (dp[i - 1, j], i - 1, j),
                (dp[i, j - 1], i, j - 1),
                (dp[i - 1, j - 1], i - 1, j - 1),
            ]
            _, i, j = min(opts, key=lambda z: z[0])
    path_rev.append((0, 0))
    path = np.array(list(reversed(path_rev)), dtype=np.int64)
    return path


def rmse_dtw_per_channel(
    y_true: np.ndarray,
    y_hat: np.ndarray,
) -> tuple[float, float]:
    """RMSE along DTW path for d (ch 0) and v (ch 1)."""
    path = _dtw_path_sqeuclidean(y_true, y_hat)
    k = path.shape[0]
    if k == 0:
        return float("nan"), float("nan")
    ii, jj = path[:, 0], path[:, 1]
    ed = y_true[ii, 0] - y_hat[jj, 0]
    ev = y_true[ii, 1] - y_hat[jj, 1]
    rmse_d = float(np.sqrt(np.mean(ed * ed)))
    rmse_v = float(np.sqrt(np.mean(ev * ev)))
    return rmse_d, rmse_v


METHOD_KEYS = ("pca_generic", "pca_personal", "vae_personal")
METHOD_LABELS = {
    "pca_generic": "Generic",
    "pca_personal": "Personalized-PCA",
    "vae_personal": "Personalized-VAE",
}


def _pool_indices(y_lab: np.ndarray, maneuver_filter: str) -> np.ndarray:
    y = np.asarray(y_lab).ravel().astype(int)
    if maneuver_filter == "all":
        return np.arange(len(y), dtype=np.int64)
    if maneuver_filter == "lane_change":
        return np.flatnonzero(np.isin(y, (MNV_LCL, MNV_LCR)))
    if maneuver_filter == "lk":
        return np.flatnonzero(y == MNV_LK)
    if maneuver_filter == "lcl":
        return np.flatnonzero(y == MNV_LCL)
    if maneuver_filter == "lcr":
        return np.flatnonzero(y == MNV_LCR)
    raise ValueError(f"Unknown maneuver filter: {maneuver_filter}")


def main() -> None:
    p = argparse.ArgumentParser(description="DTW-aligned RMSE for style evaluation trajectories.")
    p.add_argument(
        "--arrays",
        type=str,
        default=str(_REPO / "Traffic/style_eval_results/style_eval_arrays.npz"),
        help="style_eval_arrays.npz from style_learn_evaluate.py",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory for CSVs (default: same directory as --arrays)",
    )
    p.add_argument(
        "--maneuver-filter",
        type=str,
        default="all",
        choices=("all", "lane_change", "lk", "lcl", "lcr"),
        help="Which validation segments to include (default: all).",
    )
    args = p.parse_args()

    arr_path = Path(args.arrays).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else arr_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(arr_path)
    y_true = data["Y_val_true"]
    y_models = {
        "pca_generic": data["Y_pca_generic"],
        "pca_personal": data["Y_pca_personal"],
        "vae_personal": data["Y_vae_personal"],
    }
    y_va = data["y_val"]
    vid_va = data["vehicle_id_val"]
    val_idx = data["val_idx"]

    pool = _pool_indices(y_va, args.maneuver_filter)
    suffix = "" if args.maneuver_filter == "all" else f"__{args.maneuver_filter}"

    rows_long: list[dict] = []
    for pi, j in enumerate(pool):
        j = int(j)
        yt = y_true[j]
        did = int(vid_va[j])
        mnv = int(y_va[j])
        gidx = int(val_idx[j])
        for mkey in METHOD_KEYS:
            rmse_d, rmse_v = rmse_dtw_per_channel(yt, y_models[mkey][j])
            rows_long.append(
                {
                    "val_segment_index": pi,
                    "driver_id": did,
                    "maneuver": mnv,
                    "val_idx": gidx,
                    "model": mkey,
                    "method_label": METHOD_LABELS[mkey],
                    "rmse_dtw_d": rmse_d,
                    "rmse_dtw_v": rmse_v,
                }
            )

    df_long = pd.DataFrame(rows_long)
    agg_rows = []
    for mkey in METHOD_KEYS:
        sub = df_long[df_long["model"] == mkey]
        n = int(len(sub))
        std_d = float(sub["rmse_dtw_d"].std(ddof=1)) if n > 1 else 0.0
        std_v = float(sub["rmse_dtw_v"].std(ddof=1)) if n > 1 else 0.0
        se_d = std_d / np.sqrt(n) if n > 0 else float("nan")
        se_v = std_v / np.sqrt(n) if n > 0 else float("nan")
        agg_rows.append(
            {
                "model": mkey,
                "method_label": METHOD_LABELS[mkey],
                "n_segments": n,
                "rmse_dtw_d_mean": float(sub["rmse_dtw_d"].mean()),
                "rmse_dtw_d_std": std_d,
                "rmse_dtw_d_se": float(se_d),
                "rmse_dtw_v_mean": float(sub["rmse_dtw_v"].mean()),
                "rmse_dtw_v_std": std_v,
                "rmse_dtw_v_se": float(se_v),
            }
        )
    df_agg = pd.DataFrame(agg_rows)

    f_long = out_dir / f"dtw_rmse_per_segment_long{suffix}.csv"
    f_agg = out_dir / f"dtw_rmse_aggregate{suffix}.csv"
    df_long.to_csv(f_long, index=False)
    df_agg.to_csv(f_agg, index=False)
    print(f"Wrote {f_agg} ({len(df_agg)} rows)")
    print(f"Wrote {f_long} ({len(df_long)} rows)")


if __name__ == "__main__":
    main()
