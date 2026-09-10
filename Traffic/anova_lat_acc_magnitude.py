#!/usr/bin/env python3
"""
Two-way ANOVA on lateral acceleration magnitude (``lat_acc_magnitude_per_segment_long*.csv``).

Factors
-------
- **Task**: LCL, LCR, LK (``__all``) or LCL, LCR only (``__lane_change``), from ``maneuver``.
- **Method**: ``model`` — original, pca_generic, pca_personal, vae_generic, vae_personal.

Response
--------
- ``lat_acc_mag_mean``

By default, runs on **both** exports from ``export_lat_acc_magnitude_anova_csvs.py`` and
writes separate ANOVA / LS means / Tukey CSVs with suffixes ``__all`` and ``__lane_change``.

Outputs (per scope)
-------------------
- Type II ANOVA: **F** and **p** for task, method, task×method, residual.
- LS mean per **method** (marginal over task levels present in that dataset).
- Tukey HSD pairwise **method** comparisons (one-way on method; same caveat as
  ``anova_lane_change_metrics.py``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from Traffic.config import MNV_LCL, MNV_LCR, MNV_LK

TASK_ORDER_FULL = ("LCL", "LCR", "LK")
MANEUVER_TO_TASK = {
    MNV_LCL: "LCL",
    MNV_LCR: "LCR",
    MNV_LK: "LK",
}

METHOD_ORDER_FULL = (
    "original",
    "pca_generic",
    "pca_personal",
    "vae_generic",
    "vae_personal",
)

DEP = "lat_acc_mag_mean"


def _fmt_sci(x: float | None) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    return f"{float(x):.12e}"


def _prepare_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    d = df.copy()
    d["task"] = d["maneuver"].map(MANEUVER_TO_TASK)
    d = d.dropna(subset=["task"])
    d["method"] = d["model"].astype(str)
    unknown = ~d["method"].isin(METHOD_ORDER_FULL)
    if unknown.any():
        raise SystemExit(
            f"Unknown model values: {sorted(d.loc[unknown, 'method'].unique().tolist())}"
        )
    task_cats = tuple(t for t in TASK_ORDER_FULL if (d["task"] == t).any())
    method_cats = tuple(m for m in METHOD_ORDER_FULL if (d["method"] == m).any())
    d["task"] = pd.Categorical(d["task"], categories=task_cats)
    d["method"] = pd.Categorical(d["method"], categories=method_cats)
    return d, task_cats, method_cats


def _anova_table(d: pd.DataFrame, dep: str, scope: str) -> tuple[pd.DataFrame, Any]:
    dd = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[dep])
    dd = dd.copy()
    dd["y"] = dd[dep].astype(float)

    formula = "y ~ C(task) + C(method) + C(task):C(method)"
    fit = ols(formula, data=dd).fit()
    table = sm.stats.anova_lm(fit, typ=2)
    table.index.name = "Source"
    table = table.reset_index()
    table.insert(0, "scope", scope)
    table.insert(1, "dependent_variable", dep)
    table = table.rename(columns={"PR(>F)": "p_value"})

    def _rename_source(s: str) -> str:
        if s == "C(task)":
            return "task"
        if s == "C(method)":
            return "method"
        if s == "C(task):C(method)":
            return "task:method"
        return str(s)

    table["Source"] = table["Source"].map(_rename_source)
    return table, fit


def _ls_means_method(
    fit,
    task_cats: tuple[str, ...],
    method_cats: tuple[str, ...],
    dep: str,
    scope: str,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for m in method_cats:
        grid = pd.DataFrame({"task": list(task_cats), "method": [m] * len(task_cats)})
        grid["task"] = pd.Categorical(grid["task"], categories=task_cats)
        grid["method"] = pd.Categorical(grid["method"], categories=method_cats)
        pred = fit.predict(grid)
        ls = float(np.mean(pred))
        rows.append(
            {
                "scope": scope,
                "dependent_variable": dep,
                "method": m,
                "ls_mean": _fmt_sci(ls),
            }
        )
    return pd.DataFrame(rows)


def _tukey_method_table(
    y: np.ndarray, groups: np.ndarray, dep: str, scope: str
) -> pd.DataFrame:
    mask = np.isfinite(y)
    y = y[mask]
    groups = np.asarray(groups)[mask]
    if len(np.unique(groups)) < 2:
        return pd.DataFrame(
            columns=[
                "scope",
                "dependent_variable",
                "group1",
                "group2",
                "mean_diff",
                "conf_low",
                "conf_high",
                "p_tukey",
                "reject",
            ]
        )
    res = pairwise_tukeyhsd(y, groups, alpha=0.05)
    gu = list(res.groupsunique)
    out_rows: list[dict[str, str | bool]] = []
    idx = 0
    for kk in range(len(gu)):
        for jj in range(kk + 1, len(gu)):
            out_rows.append(
                {
                    "scope": scope,
                    "dependent_variable": dep,
                    "group1": str(gu[kk]),
                    "group2": str(gu[jj]),
                    "mean_diff": _fmt_sci(float(res.meandiffs[idx])),
                    "conf_low": _fmt_sci(float(res.confint[idx, 0])),
                    "conf_high": _fmt_sci(float(res.confint[idx, 1])),
                    "p_tukey": _fmt_sci(float(res.pvalues[idx])),
                    "reject": bool(res.reject[idx]),
                }
            )
            idx += 1
    return pd.DataFrame(out_rows)


def _write_anova_sci(anova: pd.DataFrame, path: Path) -> None:
    out = anova.copy()
    for col in ("sum_sq", "df", "F", "p_value"):
        if col not in out.columns:
            continue
        out[col] = out[col].map(lambda v: _fmt_sci(v) if pd.notna(v) and np.isfinite(v) else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def run_one_scope(
    inp: Path,
    out_dir: Path,
    scope: str,
) -> None:
    df_raw = pd.read_csv(inp)
    for col in ("maneuver", "model", DEP):
        if col not in df_raw.columns:
            raise SystemExit(f"Missing column {col!r} in {inp}")

    d, task_cats, method_cats = _prepare_frame(df_raw)

    table, fit = _anova_table(d, DEP, scope)
    dd = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[DEP]).copy()
    dd["y"] = dd[DEP].astype(float)
    ls_df = _ls_means_method(fit, task_cats, method_cats, DEP, scope)
    tukey_df = _tukey_method_table(
        dd["y"].values.astype(float),
        dd["method"].astype(str).values,
        DEP,
        scope,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = f"__{scope}"
    anova_path = out_dir / f"lat_acc_mag_two_way_anova{sfx}.csv"
    ls_path = out_dir / f"lat_acc_mag_lsmeans_method{sfx}.csv"
    tukey_path = out_dir / f"lat_acc_mag_tukey_hsd_method{sfx}.csv"

    _write_anova_sci(table, anova_path)
    ls_df.to_csv(ls_path, index=False)
    tukey_df.to_csv(tukey_path, index=False)

    print(f"\n=== scope={scope} ({inp.name}) ===\n")
    print(table.to_string(index=False))
    print("\n--- LS means (method, marginal over task) ---\n")
    print(ls_df.to_string(index=False))
    print("\n--- Tukey HSD (pairwise methods) ---\n")
    print(tukey_df.to_string(index=False))
    print(f"\nWrote {anova_path}", flush=True)
    print(f"Wrote {ls_path}", flush=True)
    print(f"Wrote {tukey_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="2-way ANOVA (task × method) on lat acc magnitude + LS means + Tukey."
    )
    p.add_argument(
        "--input",
        type=str,
        default="",
        help="Single long CSV (if set, only this file is processed).",
    )
    p.add_argument(
        "--scope",
        type=str,
        default="",
        help="Label for outputs when using --input (default: inferred from filename).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(_REPO / "Traffic/style_eval_results/plots"),
        help="Directory for output CSVs.",
    )
    p.add_argument(
        "--no-default-both",
        action="store_true",
        help="When combined with empty --input, do nothing (use --input).",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()

    if args.input:
        inp = Path(args.input).expanduser().resolve()
        if not inp.is_file():
            raise SystemExit(f"Input not found: {inp}")
        scope = args.scope.strip()
        if not scope:
            name = inp.stem.lower()
            scope = "lane_change" if "lane_change" in name else "all"
        run_one_scope(inp, out_dir, scope)
        return

    if args.no_default_both:
        raise SystemExit("Provide --input or omit --no-default-both.")

    plots = out_dir
    all_csv = plots / "lat_acc_magnitude_per_segment_long__all.csv"
    lc_csv = plots / "lat_acc_magnitude_per_segment_long__lane_change.csv"
    if not all_csv.is_file() or not lc_csv.is_file():
        raise SystemExit(
            f"Expected default inputs:\n  {all_csv}\n  {lc_csv}\n"
            "Run export_lat_acc_magnitude_anova_csvs.py first, or pass --input."
        )
    run_one_scope(all_csv, plots, "all")
    run_one_scope(lc_csv, plots, "lane_change")


if __name__ == "__main__":
    main()
