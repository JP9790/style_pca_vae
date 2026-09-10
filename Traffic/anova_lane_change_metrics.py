#!/usr/bin/env python3
"""
Two-way ANOVA on lane-change metrics from ``lane_change_per_segment_long*.csv``.

Factors
-------
- **Task**: maneuver type (LCL vs LCR), from column ``maneuver`` (1 / 2).
- **Method**: reconstruction model (``original``, ``pca_generic``, …).

Responses
---------
- ``lc_peak_abs_v``
- ``lc_duration_s``

Outputs
-------
- Type II ANOVA (``lane_change_two_way_anova.csv``): **F** and **p** in scientific notation.
- **LS means** for each method, marginal over task: average of fitted cell means across
  LCL and LCR for that method (``lane_change_lsmeans_method.csv``).
- **Tukey HSD** pairwise comparisons among methods on the raw response (one-way
  studentized range; **p** = adjusted p-value). See docstring caveat.

**Note:** Rows are segment×method replicates. Tukey HSD here uses the classical
one-way structure on ``method`` (pooled variance across all observations in that
response column); it does not re-use the two-way error term. For inference aligned
strictly with the two-way model, consider contrasts / emmeans with the fitted
``fit.mse_resid``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def _fmt_sci(x: float | None) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    return f"{float(x):.12e}"


def _anova_table(df: pd.DataFrame, dep: str) -> pd.DataFrame:
    d = df[[dep, "maneuver", "model"]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[dep])
    d["task"] = d["maneuver"].map({1: "LCL", 2: "LCR"})
    d["method"] = d["model"].astype(str)
    d = d.dropna(subset=["task"])
    d["y"] = d[dep].astype(float)
    task_cats = ["LCL", "LCR"]
    method_cats = sorted(d["method"].unique())
    d["task"] = pd.Categorical(d["task"], categories=task_cats)
    d["method"] = pd.Categorical(d["method"], categories=method_cats)

    formula = "y ~ C(task) + C(method) + C(task):C(method)"
    fit = ols(formula, data=d).fit()
    table = sm.stats.anova_lm(fit, typ=2)
    table.index.name = "Source"
    table = table.reset_index()
    table.insert(0, "dependent_variable", dep)
    table = table.rename(columns={"PR(>F)": "p_value"})
    return table, fit, d, task_cats, method_cats


def _ls_means_method(
    fit,
    task_cats: list[str],
    method_cats: list[str],
    dep: str,
) -> pd.DataFrame:
    """Marginal LS mean for each method = mean of fitted values over tasks."""
    rows: list[dict[str, str]] = []
    for m in method_cats:
        grid = pd.DataFrame({"task": task_cats, "method": [m] * len(task_cats)})
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
    """Pairwise Tukey HSD on ``y`` grouped by ``method`` (one-way)."""
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
    """Write ANOVA table with numeric columns as scientific strings."""
    out = anova.copy()
    for col in ("sum_sq", "df", "F", "p_value"):
        if col not in out.columns:
            continue
        out[col] = out[col].map(lambda v: _fmt_sci(v) if pd.notna(v) and np.isfinite(v) else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="2-way ANOVA + LS means + Tukey HSD (methods).")
    p.add_argument(
        "--input",
        type=str,
        default=str(
            _REPO / "Traffic/style_eval_results/plots/lane_change_per_segment_long__lane_change.csv"
        ),
        help="Long-format lane-change CSV (from plot_style_eval_results).",
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

    df = pd.read_csv(inp)
    for col in ("lc_peak_abs_v", "lc_duration_s", "maneuver", "model"):
        if col not in df.columns:
            raise SystemExit(f"Missing column {col!r} in {inp}")

    anova_parts: list[pd.DataFrame] = []
    ls_parts: list[pd.DataFrame] = []
    tukey_parts: list[pd.DataFrame] = []

    for dep in ("lc_peak_abs_v", "lc_duration_s"):
        table, fit, d, task_cats, method_cats = _anova_table(df, dep)
        anova_parts.append(table)
        ls_parts.append(_ls_means_method(fit, task_cats, method_cats, dep))
        tukey_parts.append(
            _tukey_method_table(
                d["y"].values.astype(float),
                d["method"].astype(str).values,
                dep,
            )
        )

    anova_combined = pd.concat(anova_parts, ignore_index=True)
    ls_combined = pd.concat(ls_parts, ignore_index=True)
    tukey_combined = pd.concat(tukey_parts, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    anova_path = out_dir / "lane_change_two_way_anova.csv"
    ls_path = out_dir / "lane_change_lsmeans_method.csv"
    tukey_path = out_dir / "lane_change_tukey_hsd_method.csv"

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
