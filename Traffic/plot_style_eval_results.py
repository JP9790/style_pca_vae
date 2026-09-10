#!/usr/bin/env python3
"""
Visualize NGSIM style-evaluation trajectories from ``style_learn_evaluate.py`` output.

Color convention (aligned with ``ProMP/plot_detailed_demo_generic_pca_vae.py``):
  - Black: original validation trajectory (ground truth)
  - Red: generic (PCA reconstruction, style coordinates zeroed in the style subspace)
  - Blue: personalized (PCA), thicker
  - Orange: personalized (VAE), thicker

Lane-change views (``--filter lane_change``):
  - Only LCL/LCR validation segments are sampled.
  - Extra **phase portrait**: lateral displacement vs lateral velocity (parametric in time).
  - Optional bars: peak |v| and lane-change duration proxy from ``segment_level_stats.csv``
    (LCL/LCR rows): Original, generic (PCA), Personalized-PCA, Personalized-VAE (no generic VAE bar);
    error bars = sample std over segment-level values per model.
    Same aggregates are written to ``lane_change_aggregate_stats*.csv``.
    Lateral acceleration magnitude (mean |dv/dt|) aggregates: ``lat_acc_magnitude_aggregate_stats.csv``
    (demo / generic / personalized-PCA / personalized-VAE; all validation vs lane-change-only).
    Per-segment lane-change tables: ``lane_change_per_segment_wide*.csv`` (one row per
    validation segment, peak/duration columns per model) and ``lane_change_per_segment_long*.csv``
    (one row per segment×model, all stats columns).

Reads ``style_eval_arrays.npz`` (paths configurable via CLI).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from Traffic.config import DT_S, MNV_LCL, MNV_LCR, MNV_LK

_mpl_dir = _REPO / "Traffic" / ".mplcache"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

COLOR_ORIGINAL = "black"
COLOR_GENERIC = "red"
COLOR_PCA_PERSONAL = "blue"
COLOR_VAE_PERSONAL = "orange"
LW_ORIGINAL = 2.0
LW_GENERIC = 2.0
LW_PCA = 3.5
LW_VAE = 3.5

MNV_NAMES = {MNV_LK: "LK", MNV_LCL: "LCL", MNV_LCR: "LCR"}


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


def _suffix(maneuver_filter: str) -> str:
    return "" if maneuver_filter == "all" else f"__{maneuver_filter}"


def plot_segment_phase(
    ax: plt.Axes,
    y_orig: np.ndarray,
    y_pca_gen: np.ndarray,
    y_pca_per: np.ndarray,
    y_vae_per: np.ndarray,
    title: str,
    show_legend: bool,
) -> None:
    """Phase plot: lateral d (x) vs lateral v (y); time moves along each curve."""
    series = (
        (y_orig, COLOR_ORIGINAL, LW_ORIGINAL, "Original", "-", 2),
        (y_pca_gen, COLOR_GENERIC, LW_GENERIC, "Generic (PCA)", "--", 3),
        (y_pca_per, COLOR_PCA_PERSONAL, LW_PCA, "Personalized (PCA)", "-", 4),
        (y_vae_per, COLOR_VAE_PERSONAL, LW_VAE, "Personalized (VAE)", "-", 5),
    )
    for y_arr, color, lw, lab, ls, zo in series:
        d, v = y_arr[:, 0], y_arr[:, 1]
        ax.plot(d, v, color=color, lw=lw, ls=ls, zorder=zo, label=lab)
        # Mark start (circle) and end (square) for orientation in time
        ax.scatter(d[0], v[0], color=color, s=22, marker="o", zorder=zo + 5, edgecolors="white", linewidths=0.5)
        ax.scatter(d[-1], v[-1], color=color, s=22, marker="s", zorder=zo + 5, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("lateral $d$")
    ax.set_ylabel("lateral $v$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    ax.axvline(0.0, color="0.7", lw=0.8, zorder=0)
    if show_legend:
        ax.legend(loc="best", fontsize=7, framealpha=0.9)


def plot_segment_column(
    axes_d: plt.Axes,
    axes_v: plt.Axes,
    t_s: np.ndarray,
    y_orig: np.ndarray,
    y_pca_gen: np.ndarray,
    y_pca_per: np.ndarray,
    y_vae_per: np.ndarray,
    title: str,
    show_legend: bool,
) -> None:
    """Plot lateral d (top) and lateral v (bottom) for one segment."""
    d0, v0 = y_orig[:, 0], y_orig[:, 1]
    d_g, v_g = y_pca_gen[:, 0], y_pca_gen[:, 1]
    d_p, v_p = y_pca_per[:, 0], y_pca_per[:, 1]
    d_vae, v_vae = y_vae_per[:, 0], y_vae_per[:, 1]

    # zorder + dashed generic: when gen ≈ pers-PCA, blue would hide red otherwise
    axes_d.plot(t_s, d0, color=COLOR_ORIGINAL, lw=LW_ORIGINAL, zorder=2, label="Original")
    axes_d.plot(
        t_s,
        d_g,
        color=COLOR_GENERIC,
        lw=LW_GENERIC,
        ls="--",
        zorder=3,
        label="Generic (PCA)",
    )
    axes_d.plot(t_s, d_p, color=COLOR_PCA_PERSONAL, lw=LW_PCA, zorder=4, label="Personalized (PCA)")
    axes_d.plot(t_s, d_vae, color=COLOR_VAE_PERSONAL, lw=LW_VAE, zorder=5, label="Personalized (VAE)")
    axes_d.set_ylabel("lateral $d$")
    axes_d.set_title(title)
    axes_d.grid(True, alpha=0.3)

    axes_v.plot(t_s, v0, color=COLOR_ORIGINAL, lw=LW_ORIGINAL, zorder=2, label="Original")
    axes_v.plot(
        t_s,
        v_g,
        color=COLOR_GENERIC,
        lw=LW_GENERIC,
        ls="--",
        zorder=3,
        label="Generic (PCA)",
    )
    axes_v.plot(t_s, v_p, color=COLOR_PCA_PERSONAL, lw=LW_PCA, zorder=4, label="Personalized (PCA)")
    axes_v.plot(t_s, v_vae, color=COLOR_VAE_PERSONAL, lw=LW_VAE, zorder=5, label="Personalized (VAE)")
    axes_v.set_xlabel("time (s)")
    axes_v.set_ylabel("lateral $v$")
    axes_v.grid(True, alpha=0.3)

    if show_legend:
        axes_v.legend(loc="best", fontsize=8, framealpha=0.9)


def _want_phase(maneuver_filter: str, phase: str) -> bool:
    if phase == "on":
        return True
    if phase == "off":
        return False
    return maneuver_filter in ("lane_change", "lcl", "lcr")


def _pick_segment_indices(
    n: int,
    vid: np.ndarray,
    pool: np.ndarray,
    n_segments: int,
    per_driver: bool,
    rng: np.random.Generator,
) -> list[int]:
    if len(pool) == 0:
        return []
    pool = np.asarray(pool, dtype=np.int64)
    if per_driver:
        chosen: list[int] = []
        for v in np.unique(vid[pool]):
            sub = np.intersect1d(np.flatnonzero(vid == v), pool, assume_unique=True)
            if len(sub) == 0:
                continue
            chosen.append(int(rng.choice(sub)))
        rng.shuffle(chosen)
        return chosen[:n_segments]
    if n_segments >= len(pool):
        return [int(i) for i in pool.tolist()]
    return [int(i) for i in rng.choice(pool, size=n_segments, replace=False)]


def run(
    npz_path: Path,
    out_dir: Path,
    n_segments: int,
    seed: int,
    per_driver: bool,
    dpi: int,
    stats_csv: Path | None,
    maneuver_filter: str,
    phase: str,
    segment_csv: Path | None,
) -> None:
    data = np.load(npz_path, allow_pickle=True)
    Y0 = data["Y_val_true"]
    Yg = data["Y_pca_generic"]
    Yp = data["Y_pca_personal"]
    Yv = data["Y_vae_personal"]
    y_lab = np.asarray(data["y_val"]).ravel()
    vid = np.asarray(data["vehicle_id_val"]).ravel()

    n = Y0.shape[0]
    t_s = np.arange(Y0.shape[1], dtype=np.float64) * DT_S
    rng = np.random.default_rng(seed)

    pool = _pool_indices(y_lab, maneuver_filter)
    if len(pool) == 0:
        raise SystemExit(
            f"No validation segments match maneuver filter {maneuver_filter!r}."
        )

    indices = _pick_segment_indices(n, vid, pool, n_segments, per_driver, rng)
    if not indices:
        raise SystemExit("No segments selected (empty index list).")

    sfx = _suffix(maneuver_filter)
    do_phase = _want_phase(maneuver_filter, phase)

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Grid: one column per segment, two rows (d, v) ---
    n_plot = len(indices)
    fig, axes = plt.subplots(
        2,
        n_plot,
        figsize=(max(4.0 * n_plot, 6), 6),
        sharex=True,
        squeeze=False,
    )
    for j, seg_i in enumerate(indices):
        title = f"veh {vid[seg_i]}  {MNV_NAMES.get(int(y_lab[seg_i]), '?')}"
        plot_segment_column(
            axes[0, j],
            axes[1, j],
            t_s,
            Y0[seg_i],
            Yg[seg_i],
            Yp[seg_i],
            Yv[seg_i],
            title=title,
            show_legend=(j == 0),
        )

    supt = "NGSIM validation: original vs generic (PCA) vs personalized PCA vs personalized VAE"
    if maneuver_filter != "all":
        supt += f"  [{maneuver_filter}]"
    fig.suptitle(supt, fontsize=11, y=1.02)
    fig.tight_layout()
    grid_path = out_dir / f"style_eval_trajectories_grid{sfx}.png"
    fig.savefig(grid_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {grid_path}", flush=True)

    # --- Phase portrait (d vs v): especially useful for lane changes ---
    if do_phase:
        fig2, axes_p = plt.subplots(
            1,
            n_plot,
            figsize=(max(3.8 * n_plot, 5), 4.2),
            squeeze=False,
        )
        for j, seg_i in enumerate(indices):
            title = f"{MNV_NAMES.get(int(y_lab[seg_i]), '?')}  veh {vid[seg_i]}"
            plot_segment_phase(
                axes_p[0, j],
                Y0[seg_i],
                Yg[seg_i],
                Yp[seg_i],
                Yv[seg_i],
                title=title,
                show_legend=(j == 0),
            )
        fig2.suptitle(
            "Phase portrait: lateral $d$ vs lateral $v$ (○ start, □ end)",
            fontsize=11,
            y=1.02,
        )
        fig2.tight_layout()
        phase_path = out_dir / f"style_eval_phase_d_vs_v{sfx}.png"
        fig2.savefig(phase_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig2)
        print(f"Wrote {phase_path}", flush=True)

    # --- One PNG per segment (larger, easier to read) ---
    for j, seg_i in enumerate(indices):
        fig, (ax_d, ax_v) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        title = (
            f"vehicle {vid[seg_i]}  {MNV_NAMES.get(int(y_lab[seg_i]), '?')}  "
            f"(idx {seg_i})"
        )
        plot_segment_column(
            ax_d,
            ax_v,
            t_s,
            Y0[seg_i],
            Yg[seg_i],
            Yp[seg_i],
            Yv[seg_i],
            title=title,
            show_legend=True,
        )
        fig.suptitle("Lateral trajectory (Frenet-style features)", fontsize=12, y=1.0)
        fig.tight_layout()
        one_path = out_dir / f"style_eval_segment_{seg_i:05d}{sfx}.png"
        fig.savefig(one_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {one_path}", flush=True)

        if do_phase:
            figp, axp = plt.subplots(1, 1, figsize=(5.5, 4.5))
            plot_segment_phase(
                axp,
                Y0[seg_i],
                Yg[seg_i],
                Yp[seg_i],
                Yv[seg_i],
                title=title,
                show_legend=True,
            )
            figp.suptitle("Lane-change phase: $d$ vs $v$", fontsize=11, y=0.98)
            figp.tight_layout()
            pp = out_dir / f"style_eval_phase_{seg_i:05d}{sfx}.png"
            figp.savefig(pp, dpi=dpi, bbox_inches="tight")
            plt.close(figp)
            print(f"Wrote {pp}", flush=True)

    # --- Optional: bar summary from driver CSV ---
    drv_csv = stats_csv
    if drv_csv is None:
        cand = out_dir.parent / "driver_level_mean_stats.csv"
        drv_csv = cand if cand.is_file() else npz_path.parent / "driver_level_mean_stats.csv"
    if drv_csv.is_file():
        _plot_driver_summary_bars(drv_csv, out_dir, dpi, sfx)

    # --- Segment-level CSV: lane-change plots + acceleration magnitude aggregates ---
    seg_path = segment_csv
    if seg_path is None:
        c2 = out_dir.parent / "segment_level_stats.csv"
        seg_path = c2 if c2.is_file() else npz_path.parent / "segment_level_stats.csv"
    if seg_path.is_file():
        _write_lat_acc_magnitude_aggregate(seg_path, out_dir)
    if seg_path.is_file() and maneuver_filter in ("lane_change", "lcl", "lcr"):
        _plot_lane_change_segment_bars(
            seg_path, out_dir, dpi, sfx, maneuver_filter
        )
        _write_lane_change_per_segment_csvs(seg_path, out_dir, sfx, maneuver_filter)


def _maneuver_matches_filter(m: int, maneuver_filter: str) -> bool:
    if maneuver_filter == "lane_change":
        return m in (MNV_LCL, MNV_LCR)
    if maneuver_filter == "lcl":
        return m == MNV_LCL
    if maneuver_filter == "lcr":
        return m == MNV_LCR
    return False


def _write_lane_change_per_segment_csvs(
    seg_csv: Path,
    out_dir: Path,
    sfx: str,
    maneuver_filter: str,
) -> None:
    """
    Export lane-change peak |v| and duration for every segment and model.

    Reads the **full** ``segment_level_stats.csv`` (all maneuvers) so row index ``j``
    matches across models, then filters by maneuver.

    - **Wide**: one row per ``j`` where maneuver matches filter; columns
      ``lc_peak_abs_v__<model>``, ``lc_duration_s__<model>`` for each model.
    - **Long**: for each such ``j``, five rows (one per model) with all stat columns.
    """
    import csv
    from collections import defaultdict

    models_order = [
        "original",
        "pca_generic",
        "pca_personal",
        "vae_generic",
        "vae_personal",
    ]
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)

    with seg_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_model[row["model"]].append(dict(row))

    lists_mod = [by_model.get(m, []) for m in models_order]
    if not lists_mod[0]:
        print(f"Skip lane-change per-segment CSVs: empty {seg_csv}", flush=True)
        return
    n0 = len(lists_mod[0])
    if not all(len(by_model.get(m, [])) == n0 for m in models_order):
        print(
            "Warning: lane_change_per_segment CSVs skipped (unequal row counts per model).",
            flush=True,
        )
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows: list[dict[str, str]] = []
    wide_fieldnames = [
        "val_segment_index",
        "driver_id",
        "maneuver",
        "maneuver_name",
    ]
    for mname in models_order:
        wide_fieldnames.append(f"lc_peak_abs_v__{mname}")
        wide_fieldnames.append(f"lc_duration_s__{mname}")
    wide_rows: list[dict[str, str | int]] = []

    for j in range(n0):
        r0 = lists_mod[0][j]
        try:
            m = int(r0.get("maneuver", -1))
        except (TypeError, ValueError):
            continue
        if not _maneuver_matches_filter(m, maneuver_filter):
            continue
        for mname in models_order:
            long_rows.append(dict(by_model[mname][j]))
        wrow: dict[str, str | int] = {
            "val_segment_index": j,
            "driver_id": r0.get("driver_id", ""),
            "maneuver": m,
            "maneuver_name": MNV_NAMES.get(m, str(m)),
        }
        for mname in models_order:
            rj = by_model[mname][j]
            wrow[f"lc_peak_abs_v__{mname}"] = rj.get("lc_peak_abs_v", "")
            wrow[f"lc_duration_s__{mname}"] = rj.get("lc_duration_s", "")
        wide_rows.append(wrow)

    long_path = out_dir / f"lane_change_per_segment_long{sfx}.csv"
    if long_rows:
        fieldnames = list(long_rows[0].keys())
        with long_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in long_rows:
                writer.writerow(row)
        print(f"Wrote {long_path} ({len(long_rows)} rows)", flush=True)
    else:
        print(f"Skip {long_path}: no segments match {maneuver_filter!r}", flush=True)

    wide_path = out_dir / f"lane_change_per_segment_wide{sfx}.csv"
    if wide_rows:
        with wide_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=wide_fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in wide_rows:
                writer.writerow(row)
        print(f"Wrote {wide_path} ({len(wide_rows)} rows)", flush=True)


def _write_lat_acc_magnitude_aggregate(seg_csv: Path, out_dir: Path) -> None:
    """
    Mean and sample std of ``lat_acc_mag_mean`` per reconstruction method.

    Rows: demo (original), generic (``pca_generic``), personalized-PCA (``pca_personal``),
    personalized-VAE (``vae_personal``). Two scopes: all validation segments, and
    lane-change-only (LCL/LCR).
    """
    import csv

    model_labels = {
        "original": "demo",
        "pca_generic": "generic",
        "pca_personal": "personalized-PCA",
        "vae_personal": "personalized-VAE",
    }
    scopes: tuple[tuple[str, tuple[int, ...] | None], ...] = (
        ("all_validation", None),
        ("lane_change_only", (MNV_LCL, MNV_LCR)),
    )

    by_scope_model: dict[str, dict[str, list[float]]] = {
        name: {k: [] for k in model_labels} for name, _ in scopes
    }

    with seg_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mnv = int(row.get("maneuver", -1))
            except (TypeError, ValueError):
                continue
            mk = row.get("model", "")
            if mk not in model_labels:
                continue
            try:
                val = float(row["lat_acc_mag_mean"])
            except (TypeError, ValueError, KeyError):
                continue
            if not np.isfinite(val):
                continue
            for scope_name, mset in scopes:
                if mset is None or mnv in mset:
                    by_scope_model[scope_name][mk].append(val)

    out_rows: list[dict[str, str | float | int]] = []
    for scope_name, _ in scopes:
        for model_key, label in model_labels.items():
            xs = by_scope_model[scope_name][model_key]
            n = len(xs)
            if n == 0:
                mu = float("nan")
                sig = float("nan")
            elif n == 1:
                mu = float(xs[0])
                sig = 0.0
            else:
                a = np.asarray(xs, dtype=np.float64)
                mu = float(np.mean(a))
                sig = float(np.std(a, ddof=1))
            out_rows.append(
                {
                    "scope": scope_name,
                    "model": model_key,
                    "model_label": label,
                    "mean_lat_acc_mag": mu,
                    "std_lat_acc_mag": sig,
                    "n_segments": n,
                    "source_segment_csv": str(seg_csv),
                }
            )

    out_path = out_dir / "lat_acc_magnitude_aggregate_stats.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "model",
        "model_label",
        "mean_lat_acc_mag",
        "std_lat_acc_mag",
        "n_segments",
        "source_segment_csv",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)
    print(f"Wrote {out_path}", flush=True)


def _plot_lane_change_segment_bars(
    seg_csv: Path,
    out_dir: Path,
    dpi: int,
    sfx: str,
    maneuver_filter: str,
) -> None:
    """Bars for LC rows: mean ± sample std (over segments) for peak |v| and duration proxy."""
    import csv

    rows: list[dict[str, str]] = []
    with seg_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                m = int(row.get("maneuver", -1))
            except (TypeError, ValueError):
                continue
            if maneuver_filter == "lcl" and m != MNV_LCL:
                continue
            if maneuver_filter == "lcr" and m != MNV_LCR:
                continue
            if maneuver_filter == "lane_change" and m not in (MNV_LCL, MNV_LCR):
                continue
            rows.append(row)
    if not rows:
        return

    models = ["original", "pca_generic", "pca_personal", "vae_personal"]
    stats = ("lc_peak_abs_v", "lc_duration_s")
    n_lc_seg = sum(1 for row in rows if row.get("model") == "original")

    def mean_std_stat(model: str, key: str) -> tuple[float, float, int]:
        xs: list[float] = []
        for row in rows:
            if row.get("model") != model:
                continue
            try:
                xs.append(float(row[key]))
            except (TypeError, ValueError):
                continue
        if not xs:
            return float("nan"), float("nan"), 0
        a = np.asarray(xs, dtype=np.float64)
        fin = np.isfinite(a)
        n = int(np.sum(fin))
        m = float(np.nanmean(a))
        if n > 1:
            s = float(np.nanstd(a, ddof=1))
        else:
            s = 0.0
        return m, s, n

    xlabels = ["Original", "generic", "Personalized-PCA", "Personalized-VAE"]
    csv_rows: list[dict[str, str | float | int]] = []
    for mi, m in enumerate(models):
        mu_p, sig_p, n_p = mean_std_stat(m, "lc_peak_abs_v")
        mu_d, sig_d, n_d = mean_std_stat(m, "lc_duration_s")
        csv_rows.append(
            {
                "maneuver_filter": maneuver_filter,
                "n_validation_lc_segments": n_lc_seg,
                "model": m,
                "model_label": xlabels[mi],
                "lc_peak_abs_v_mean": mu_p,
                "lc_peak_abs_v_std": sig_p,
                "lc_peak_abs_v_n": n_p,
                "lc_duration_s_mean": mu_d,
                "lc_duration_s_std": sig_d,
                "lc_duration_s_n": n_d,
                "source_segment_csv": str(seg_csv),
            }
        )
    csv_path = out_dir / f"lane_change_aggregate_stats{sfx}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "maneuver_filter",
            "n_validation_lc_segments",
            "model",
            "model_label",
            "lc_peak_abs_v_mean",
            "lc_peak_abs_v_std",
            "lc_peak_abs_v_n",
            "lc_duration_s_mean",
            "lc_duration_s_std",
            "lc_duration_s_n",
            "source_segment_csv",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"Wrote {csv_path}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    x = np.arange(4)
    w = 0.6
    colors_4 = [
        COLOR_ORIGINAL,
        COLOR_GENERIC,
        COLOR_PCA_PERSONAL,
        COLOR_VAE_PERSONAL,
    ]
    for ax, stat_key, ylab in zip(
        axes,
        stats,
        ("mean peak |lateral v|", "mean LC duration proxy (s)"),
    ):
        heights: list[float] = []
        errs: list[float] = []
        for m in models:
            mu, sig, _n = mean_std_stat(m, stat_key)
            heights.append(mu)
            errs.append(sig if np.isfinite(sig) else 0.0)
        ax.bar(
            x,
            heights,
            width=w,
            color=colors_4,
            yerr=errs,
            capsize=3.5,
            ecolor="0.25",
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=7.5)
        ax.set_ylabel(f"{ylab} (mean ± std over segments)")
        ax.set_title(stat_key)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        f"Lane-change metrics ({maneuver_filter}), n={n_lc_seg} validation segments",
        fontsize=10,
    )
    fig.tight_layout()
    p = out_dir / f"lane_change_stats_bars{sfx}.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p}", flush=True)


def _plot_driver_summary_bars(drv_csv: Path, out_dir: Path, dpi: int, sfx: str) -> None:
    import csv

    rows: list[dict[str, str]] = []
    with drv_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        return

    models = ["original", "pca_generic", "pca_personal", "vae_generic", "vae_personal"]
    stat = "v_mean_time"
    # Aggregate per model (mean over drivers)
    vals: dict[str, float] = {}
    for m in models:
        xs = [float(r[stat]) for r in rows if r.get("model") == m and r.get(stat, "")]
        if xs:
            vals[m] = float(np.mean(xs))
    if len(vals) < 2:
        return

    order = ["pca_generic", "pca_personal", "vae_personal"]
    order = [k for k in order if k in vals]
    if not order:
        return
    heights = [vals[k] for k in order]
    colors = [COLOR_GENERIC, COLOR_PCA_PERSONAL, COLOR_VAE_PERSONAL]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(order))
    ax.bar(x, heights, color=colors[: len(order)], align="center")
    ax.set_xticks(x)
    ax.set_xticklabels(["Generic (PCA)", "Pers. (PCA)", "Pers. (VAE)"])
    ax.set_ylabel("mean lateral velocity (mean over drivers)")
    ax.set_title(f"Aggregate: {stat} (validation reconstructions)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"summary_bar_v_mean{sfx}.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot style_eval_arrays.npz trajectories.")
    p.add_argument(
        "--npz",
        type=str,
        default=str(_REPO / "Traffic/style_eval_results/style_eval_arrays.npz"),
        help="Path to style_eval_arrays.npz",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(_REPO / "Traffic/style_eval_results/plots"),
        help="Directory for PNG output",
    )
    p.add_argument("--n-segments", type=int, default=6, help="Number of segments to plot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--per-driver",
        action="store_true",
        help="Pick one random segment per driver (up to n-segments drivers)",
    )
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--stats-csv",
        type=str,
        default=None,
        help="Optional driver_level_mean_stats.csv for summary bar chart.",
    )
    p.add_argument(
        "--filter",
        type=str,
        choices=("all", "lane_change", "lk", "lcl", "lcr"),
        default="all",
        help="Restrict plotted segments by maneuver (lane_change = LCL or LCR).",
    )
    p.add_argument(
        "--phase",
        type=str,
        choices=("auto", "on", "off"),
        default="auto",
        help="Phase plots d vs v: auto=on for lane-change filters, off otherwise.",
    )
    p.add_argument(
        "--segment-csv",
        type=str,
        default=None,
        help="segment_level_stats.csv for lane-change bar charts (default: next to npz).",
    )
    args = p.parse_args()

    npz_path = Path(args.npz).expanduser().resolve()
    if not npz_path.exists():
        raise SystemExit(f"Not found: {npz_path}")

    sc = Path(args.stats_csv).expanduser().resolve() if args.stats_csv else None
    segc = Path(args.segment_csv).expanduser().resolve() if args.segment_csv else None

    run(
        npz_path,
        Path(args.out_dir).expanduser().resolve(),
        n_segments=args.n_segments,
        seed=args.seed,
        per_driver=args.per_driver,
        dpi=args.dpi,
        stats_csv=sc,
        maneuver_filter=args.filter,
        phase=args.phase,
        segment_csv=segc,
    )


if __name__ == "__main__":
    main()
