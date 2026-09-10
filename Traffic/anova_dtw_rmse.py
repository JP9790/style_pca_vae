#!/usr/bin/env python3
"""
Two-way ANOVA on DTW-aligned RMSE (``dtw_rmse_per_segment_long*.csv``).

Factors
-------
- **Task**: LCL, LCR, LK from column ``maneuver`` (1 / 2 / 0 in NGSIM encoding).
- **Method**: Generic (~ ``pca_generic``), PCA (~ ``pca_personal``), VAE (~ ``vae_personal``).

Responses
---------
- ``rmse_dtw_d``, ``rmse_dtw_v``

Outputs
-------
- Type II ANOVA per response: **F** and **p** for task, method, interaction, residual.
- LS mean per **method** (marginal over task): mean of fitted cell means across LCL, LCR, LK.
- Tukey HSD pairwise comparisons among **methods** on the raw response (same caveat as
  ``anova_lane_change_metrics.py``: one-way structure on method, not the two-way error term).

Use the **all-maneuvers** long CSV from ``dtw_rmse_eval.py`` (not ``__lane_change`` only),
so all three task levels are present.
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

# User-facing task order: LCL, LCR, LK
TASK_ORDER = ("LCL", "LCR", "LK")
MANEUVER_TO_TASK = {
    MNV_LCL: "LCL",
    MNV_LCR: "LCR",
    MNV_LK: "LK",
}

# Method factor: display names used in Tukey / LS means
METHOD_MODEL_TO_LABEL = {
    "pca_generic": "Generic",
    "pca_personal": "PCA",
    "vae_personal": "VAE",
}
METHOD_ORDER = ("Generic", "PCA", "VAE")

DEP_VARS = ("rmse_dtw_d", "rmse_dtw_v")


def _fmt_sci(x: float | None) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    return f"{float(x):.12e}"


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["task"] = d["maneuver"].map(MANEUVER_TO_TASK)
    d = d.dropna(subset=["task"])
    d["method"] = d["model"].map(METHOD_MODEL_TO_LABEL)
    bad = d["method"].isna()
    if bad.any():
        raise SystemExit(
            f"Unknown model values (expected pca_generic, pca_personal, vae_personal): "
            f"{d.loc[bad, 'model'].unique().tolist()}"
        )
    d["task"] = pd.Categorical(d["task"], categories=TASK_ORDER)
    d["method"] = pd.Categorical(d["method"], categories=METHOD_ORDER)
    return d


def _anova_table(d: pd.DataFrame, dep: str) -> tuple[pd.DataFrame, Any]:
    """Type II ANOVA; ``d`` must have ``task``, ``method``, and ``dep``."""
    dd = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[dep])
    dd = dd.copy()
    dd["y"] = dd[dep].astype(float)

    formula = "y ~ C(task) + C(method) + C(task):C(method)"
    fit = ols(formula, data=dd).fit()
    table = sm.stats.anova_lm(fit, typ=2)
    table.index.name = "Source"
    table = table.reset_index()
    table.insert(0, "dependent_variable", dep)
    table = table.rename(columns={"PR(>F)": "p_value"})
    # Friendlier source labels
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
) -> pd.DataFrame:
    """Marginal LS mean for each method = mean of fitted cell means over tasks."""
    rows: list[dict[str, str]] = []
    for m in method_cats:
        grid = pd.DataFrame({"task": list(task_cats), "method": [m] * len(task_cats)})
        grid["task"] = pd.Categorical(grid["task"], categories=task_cats)
        grid["method"] = pd.Categorical(grid["method"], categories=method_cats)
        pred = fit.predict(grid)
        ls = float(np.mean(pred))
        rows.append(
            {
                "dependent_variable": dep,
                "method": m,
                "ls_mean": _fmt_sci(ls),
            }
        )
    return pd.DataFrame(rows)


def _tukey_method_table(y: np.ndarray, groups: np.ndarray, dep: str) -> pd.DataFrame:
    mask = np.isfinite(y)
    y = y[mask]
    groups = np.asarray(groups)[mask]
    if len(np.unique(groups)) < 2:
        return pd.DataFrame(
            columns=[
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
    pairs: list[tuple[str, str]] = []
    for i in range(len(gu)):
        for j in range(i + 1, len(gu)):
            pairs.append((str(gu[i]), str(gu[j])))
    out_rows: list[dict[str, str | bool]] = []
    for k, (g1, g2) in enumerate(pairs):
        out_rows.append(
            {
                "dependent_variable": dep,
                "group1": g1,
                "group2": g2,
                "mean_diff": _fmt_sci(float(res.meandiffs[k])),
                "conf_low": _fmt_sci(float(res.confint[k, 0])),
                "conf_high": _fmt_sci(float(res.confint[k, 1])),
                "p_tukey": _fmt_sci(float(res.pvalues[k])),
                "reject": bool(res.reject[k]),
            }
        )
    return pd.DataFrame(out_rows)


def _write_anova_sci(anova: pd.DataFrame, path: Path) -> None:
    out = anova.copy()
    for col in ("sum_sq", "df", "F", "p_value"):
        if col not in out.columns:
            continue
        out[col] = out[col].map(lambda v: _fmt_sci(v) if pd.notna(v) and np.isfinite(v) else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main() -> None:
    p = argparse.ArgumentParser(
        description="2-way ANOVA (task × method) on DTW RMSE + LS means + Tukey HSD."
    )
    p.add_argument(
        "--input",
        type=str,
        default=str(_REPO / "Traffic/style_eval_results/dtw_rmse_per_segment_long.csv"),
        help="Long CSV from dtw_rmse_eval.py (all maneuvers).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(_REPO / "Traffic/style_eval_results/plots"),
        help="Directory for output CSVs.",
    )
    args = p.parse_args()

    inp = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    df_raw = pd.read_csv(inp)
    for col in ("maneuver", "model", "rmse_dtw_d", "rmse_dtw_v"):
        if col not in df_raw.columns:
            raise SystemExit(f"Missing column {col!r} in {inp}")

    d = _prepare_frame(df_raw)

    anova_parts: list[pd.DataFrame] = []
    ls_parts: list[pd.DataFrame] = []
    tukey_parts: list[pd.DataFrame] = []

    for dep in DEP_VARS:
        table, fit = _anova_table(d, dep)
        anova_parts.append(table)
        dd = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[dep]).copy()
        dd["y"] = dd[dep].astype(float)
        ls_parts.append(_ls_means_method(fit, TASK_ORDER, METHOD_ORDER, dep))
        tukey_parts.append(
            _tukey_method_table(
                dd["y"].values.astype(float),
                dd["method"].astype(str).values,
                dep,
            )
        )

    anova_combined = pd.concat(anova_parts, ignore_index=True)
    ls_combined = pd.concat(ls_parts, ignore_index=True)
    tukey_combined = pd.concat(tukey_parts, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    anova_path = out_dir / "dtw_rmse_two_way_anova.csv"
    ls_path = out_dir / "dtw_rmse_lsmeans_method.csv"
    tukey_path = out_dir / "dtw_rmse_tukey_hsd_method.csv"

    _write_anova_sci(anova_combined, anova_path)
    ls_combined.to_csv(ls_path, index=False)
    tukey_combined.to_csv(tukey_path, index=False)

    print(anova_combined.to_string(index=False))
    print("\n--- LS means (method, marginal over task) ---\n")
    print(ls_combined.to_string(index=False))
    print("\n--- Tukey HSD (pairwise methods) ---\n")
    print(tukey_combined.to_string(index=False))
    print(f"\nWrote {anova_path}", flush=True)
    print(f"Wrote {ls_path}", flush=True)
    print(f"Wrote {tukey_path}", flush=True)


if __name__ == "__main__":
    main()
