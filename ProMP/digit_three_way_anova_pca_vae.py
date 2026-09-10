#!/usr/bin/env python3
"""
Three-way ANOVA for digit metrics: Task × Pipeline (PCA vs VAE) × Condition (Generic vs Personalized).

Merges:
  - PCA-based all-user digit evaluation (default: all_users_digit_evaluation.csv)
  - VAE-based all-user digit evaluation (default: all_users_digit_vae_l2_iou.csv)

Each file provides, per (user, digit), mean L2 and mean IoU for generic vs personalized libraries.
Rows are stacked into long form with factors:
  - task: digit 0–9
  - pipeline: PCA | VAE
  - condition: generic | personalized

The model controls for between-user heterogeneity with fixed effects C(user_id), then tests
Type III sums of squares for task, pipeline, condition, and all interactions (statsmodels).

Also reports:
  - Unadjusted marginal means by condition (and by pipeline × condition)
  - Tukey HSD for pairwise comparisons (exploratory; observations are repeated within user)

Outputs: JSON (machine-readable F, p, df, marginal means) and a text file with LaTeX-friendly lines.

Example narrative (fill from output):
  For digit writing, significant main effects were found for Task ($F = ...$, $p = ...$),
  Pipeline ($F = ...$, $p = ...$), and Condition / personalization ($F = ...$, $p = ...$).
  ...
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from statsmodels.formula.api import ols
    import statsmodels.api as sm
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This script requires statsmodels. Install with: pip install statsmodels pandas"
    ) from e

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
except ImportError:  # pragma: no cover
    pairwise_tukeyhsd = None


def _latex_scientific(p: float) -> str:
    if p == 0.0 or (p > 0 and p < 1e-300):
        return r"$p < 10^{-300}$"
    if p >= 0.001:
        return f"$p = {p:.4g}$"
    s = f"{p:.2e}"
    m = re.match(r"([\d.]+)e([+-]\d+)", s)
    if not m:
        return f"$p = {p}$"
    mant, exp = m.group(1), int(m.group(2))
    return rf"$p = {mant}\times 10^{{{exp}}}$"


def _fmt_f(fv: float) -> str:
    return f"{fv:.4f}".rstrip("0").rstrip(".")


def load_pca_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {
        "User": "user_id",
        "Task": "task",
        "L2_Generic": "l2_generic",
        "L2_Personalized": "l2_personalized",
        "IoU_Generic": "iou_generic",
        "IoU_Personalized": "iou_personalized",
    }
    df = df.rename(columns=rename)
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        uid = int(r["user_id"])
        task = str(int(r["task"]))
        rows.append(
            {
                "user_id": uid,
                "task": task,
                "pipeline": "PCA",
                "condition": "generic",
                "L2": float(r["l2_generic"]),
                "IoU": float(r["iou_generic"]),
            }
        )
        rows.append(
            {
                "user_id": uid,
                "task": task,
                "pipeline": "PCA",
                "condition": "personalized",
                "L2": float(r["l2_personalized"]),
                "IoU": float(r["iou_personalized"]),
            }
        )
    return pd.DataFrame(rows)


def load_vae_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # tolerate alternate filename typo
    cols = {c.lower().replace(" ", "_"): c for c in df.columns}
    def col(*names: str) -> str:
        for n in names:
            if n in df.columns:
                return n
            if n in cols:
                return cols[n]
        raise KeyError(f"Missing column among {names}")

    c_user = col("user_id", "User")
    c_digit = col("digit", "Task")
    c_l2g = col("l2_generic_mean", "L2_Generic")
    c_l2p = col("l2_personalized_mean", "L2_Personalized")
    c_ioug = col("iou_generic_mean", "IoU_Generic")
    c_ioup = col("iou_personalized_mean", "IoU_Personalized")

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        if pd.isna(r[c_l2g]) or r[c_l2g] == "":
            continue
        uid = int(r[c_user])
        task = str(int(float(r[c_digit])))
        rows.append(
            {
                "user_id": uid,
                "task": task,
                "pipeline": "VAE",
                "condition": "generic",
                "L2": float(r[c_l2g]),
                "IoU": float(r[c_ioug]),
            }
        )
        rows.append(
            {
                "user_id": uid,
                "task": task,
                "pipeline": "VAE",
                "condition": "personalized",
                "L2": float(r[c_l2p]),
                "IoU": float(r[c_ioup]),
            }
        )
    return pd.DataFrame(rows)


def build_long_table(pca_csv: Path, vae_csv: Path) -> pd.DataFrame:
    pca = load_pca_long(pca_csv)
    vae = load_vae_long(vae_csv)
    long_df = pd.concat([pca, vae], ignore_index=True)
    for col in ("user_id", "task", "pipeline", "condition"):
        long_df[col] = long_df[col].astype("category")
    long_df = long_df.dropna(subset=["L2", "IoU"])
    return long_df


def run_anova_typ3(
    df: pd.DataFrame, dep: str
) -> Tuple[pd.DataFrame, Any]:
    """
    y ~ C(user_id) + C(task) * C(pipeline) * C(condition)
    Type III tests for all terms except intercept (user effects absorbed).
    """
    formula = (
        f"{dep} ~ C(user_id) + C(task) * C(pipeline) * C(condition)"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        model = ols(formula, data=df).fit()
    table = sm.stats.anova_lm(model, typ=3)
    return table, model


def anova_table_to_records(table: pd.DataFrame) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for term in table.index:
        if term == "Residual":
            continue
        row = table.loc[term]
        out.append(
            {
                "term": str(term),
                "sum_sq": float(row["sum_sq"]),
                "df": float(row["df"]),
                "F": float(row["F"]) if pd.notna(row["F"]) else None,
                "PR(>F)": float(row["PR(>F)"]) if pd.notna(row["PR(>F)"]) else None,
            }
        )
    return out


def residual_df_from_table(table: pd.DataFrame) -> int:
    if "Residual" in table.index:
        return int(round(float(table.loc["Residual", "df"])))
    return -1


def term_from_records(records: List[Dict[str, Any]], exact: str) -> Dict[str, Any] | None:
    by_t = {r["term"]: r for r in records}
    return by_t.get(exact)


def marginal_means(df: pd.DataFrame, dep: str) -> Dict[str, Any]:
    g_cond = df.groupby("condition", observed=False)[dep].agg(["mean", "std", "count"])
    g_pipe_cond = df.groupby(["pipeline", "condition"], observed=False)[dep].agg(
        ["mean", "std", "count"]
    )
    return {
        "by_condition": g_cond.reset_index().to_dict(orient="records"),
        "by_pipeline_condition": g_pipe_cond.reset_index().to_dict(orient="records"),
    }


def tukey_condition(df: pd.DataFrame, dep: str) -> Dict[str, Any] | None:
    if pairwise_tukeyhsd is None:
        return None
    res = pairwise_tukeyhsd(
        endog=df[dep].values,
        groups=df["condition"].astype(str).values,
        alpha=0.05,
    )
    return {"summary": str(res), "pvalue_generic_vs_personalized": float(res.pvalues[0])}


def tukey_pipeline(df: pd.DataFrame, dep: str) -> Dict[str, Any] | None:
    if pairwise_tukeyhsd is None:
        return None
    res = pairwise_tukeyhsd(
        endog=df[dep].values,
        groups=df["pipeline"].astype(str).values,
        alpha=0.05,
    )
    return {"summary": str(res), "pvalue_pca_vs_vae": float(res.pvalues[0])}


def prose_lines_metric(
    records: List[Dict[str, Any]],
    mm: Dict[str, Any],
    df_resid: int,
    metric_label: str,
    lower_is_better: bool,
) -> List[str]:
    """LaTeX-style lines for one dependent variable."""
    lines: List[str] = []

    t_task = term_from_records(records, "C(task)")
    t_pipe = term_from_records(records, "C(pipeline)")
    t_cond = term_from_records(records, "C(condition)")
    t_txp = term_from_records(records, "C(task):C(pipeline)")
    t_txc = term_from_records(records, "C(task):C(condition)")
    t_pxc = term_from_records(records, "C(pipeline):C(condition)")
    t_3 = term_from_records(records, "C(task):C(pipeline):C(condition)")

    def line_effect(name: str, tr: Dict[str, Any] | None) -> None:
        if tr is None or tr["F"] is None:
            return
        dfn = int(round(tr["df"]))
        lines.append(
            f"{name}: $F = {_fmt_f(tr['F'])}$, {_latex_scientific(tr['PR(>F)'])} "
            f"(df$_1 = {dfn}$, df$_2 = {df_resid}$)"
        )

    line_effect("Task", t_task)
    line_effect("Pipeline (PCA vs VAE)", t_pipe)
    line_effect("Condition (Generic vs Personalized)", t_cond)
    line_effect("Task × Pipeline", t_txp)
    line_effect("Task × Condition", t_txc)
    line_effect("Pipeline × Condition", t_pxc)
    line_effect("Task × Pipeline × Condition", t_3)

    by_c = {str(x["condition"]): x["mean"] for x in mm["by_condition"]}
    if "generic" in by_c and "personalized" in by_c:
        g, p = by_c["generic"], by_c["personalized"]
        lines.append(
            f"Unadjusted marginal means ({metric_label}): Generic ${g:.2f}$, Personalized ${p:.2f}$."
        )
        if lower_is_better:
            if p < g:
                lines.append(
                    "Personalized shows lower mean L2 than Generic in the raw marginals "
                    "(direction consistent with reduced reconstruction error)."
                )
            else:
                lines.append(
                    "Raw marginal means do not show lower L2 for Personalized; "
                    "check interactions (e.g. Pipeline × Condition)."
                )
        else:
            if p > g:
                lines.append(
                    "Personalized shows higher mean IoU than Generic in the raw marginals "
                    "(direction consistent with improved overlap)."
                )
            else:
                lines.append(
                    "Raw marginal means favor Generic on IoU; interpret with interaction terms."
                )

    return lines


def narrative_paragraph_l2(
    records: List[Dict[str, Any]], df_resid: int, alpha: float = 0.05
) -> str:
    """Single paragraph in the spirit of the user's example (L2)."""
    t_task = term_from_records(records, "C(task)")
    t_pipe = term_from_records(records, "C(pipeline)")
    t_cond = term_from_records(records, "C(condition)")
    t_txc = term_from_records(records, "C(task):C(condition)")
    t_pxc = term_from_records(records, "C(pipeline):C(condition)")

    def sig(tr: Dict[str, Any] | None) -> bool:
        return (
            tr is not None
            and tr.get("PR(>F)") is not None
            and float(tr["PR(>F)"]) < alpha
        )

    parts: List[str] = []
    parts.append("For digit writing (L2 distance; controlling for user fixed effects), ")
    mains: List[str] = []
    for label, tr in [
        ("Task", t_task),
        ("Pipeline (PCA vs VAE)", t_pipe),
        ("Personalization (Generic vs Personalized)", t_cond),
    ]:
        if tr and tr.get("F") is not None:
            star = "significant " if sig(tr) else ""
            mains.append(
                f"{star}{label} ($F = {_fmt_f(tr['F'])}$, {_latex_scientific(tr['PR(>F)'])}, "
                f"df$_1={int(round(tr['df']))}$, df$_2={df_resid}$)"
            )
    if mains:
        parts.append("main effects: " + "; ".join(mains) + ". ")
    if t_txc and t_txc.get("F") is not None:
        lab = "significant " if sig(t_txc) else ""
        parts.append(
            f"The {lab}Task × Personalization interaction was "
            f"$F = {_fmt_f(t_txc['F'])}$, {_latex_scientific(t_txc['PR(>F)'])}. "
        )
    if t_pxc and t_pxc.get("F") is not None:
        lab = "significant " if sig(t_pxc) else ""
        parts.append(
            f"The {lab}Pipeline × Personalization interaction was "
            f"$F = {_fmt_f(t_pxc['F'])}$, {_latex_scientific(t_pxc['PR(>F)'])}. "
        )
    return "".join(parts).strip()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Three-way ANOVA: Task × Pipeline (PCA/VAE) × Condition (generic/personalized)."
    )
    ap.add_argument(
        "--pca_csv",
        type=str,
        default="ProMP/all_users_digit_evaluation.csv",
    )
    ap.add_argument(
        "--vae_csv",
        type=str,
        default="ProMP/all_users_digit_vae_l2_iou.csv",
    )
    ap.add_argument(
        "--out_json",
        type=str,
        default="ProMP/digit_three_way_anova_pca_vae.json",
    )
    ap.add_argument(
        "--out_txt",
        type=str,
        default="ProMP/digit_three_way_anova_pca_vae_summary.txt",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for labeling effects in the narrative paragraph.",
    )
    args = ap.parse_args()

    root = Path(".").resolve()
    pca_path = Path(args.pca_csv).expanduser()
    vae_path = Path(args.vae_csv).expanduser()
    if not pca_path.is_absolute():
        pca_path = root / pca_path
    if not vae_path.is_absolute():
        vae_path = root / vae_path

    long_df = build_long_table(pca_path, vae_path)
    n_users = long_df["user_id"].nunique()
    n_rows = len(long_df)

    results: Dict[str, Any] = {
        "alpha": float(args.alpha),
        "n_rows": n_rows,
        "n_users": int(n_users),
        "pca_csv": str(pca_path),
        "vae_csv": str(vae_path),
        "formula_note": (
            "OLS: y ~ C(user_id) + C(task) * C(pipeline) * C(condition); "
            "Type III ANOVA; user fixed effects control between-subject heterogeneity."
        ),
    }

    for dep, key in [("L2", "l2"), ("IoU", "iou")]:
        table, model = run_anova_typ3(long_df, dep)
        recs = anova_table_to_records(table)
        df_resid = residual_df_from_table(table)
        mm = marginal_means(long_df, dep)
        entry: Dict[str, Any] = {
            "anova_typ3": recs,
            "anova_table_csv": table.to_csv(),
            "marginal_means": mm,
            "residual_df": df_resid,
            "r_squared": float(model.rsquared),
            "r_squared_adj": float(model.rsquared_adj),
            "tukey_condition": tukey_condition(long_df, dep),
            "tukey_pipeline": tukey_pipeline(long_df, dep),
        }
        if dep == "L2":
            entry["narrative_paragraph_l2"] = narrative_paragraph_l2(
                recs, df_resid, alpha=args.alpha
            )
            tc = term_from_records(recs, "C(condition)")
            if tc and tc["PR(>F)"] is not None:
                entry["personalization_main_effect_p"] = tc["PR(>F)"]
        results[key] = entry

    out_json = Path(args.out_json).expanduser()
    if not out_json.is_absolute():
        out_json = root / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("Three-way ANOVA (digit metrics): Task × Pipeline (PCA vs VAE) × Condition")
    lines.append(f"N observations (long format): {n_rows}, users: {n_users}")
    lines.append("")
    lines.append("--- L2 (lower is better) ---")
    lines.append(results["l2"].get("narrative_paragraph_l2", ""))
    lines.append("")
    dr = int(results["l2"]["residual_df"])
    lines.extend(
        prose_lines_metric(
            results["l2"]["anova_typ3"],
            results["l2"]["marginal_means"],
            dr,
            "L2",
            lower_is_better=True,
        )
    )
    tc = term_from_records(results["l2"]["anova_typ3"], "C(condition)")
    if tc and tc.get("PR(>F)") is not None:
        lines.append(
            "Type III F-test for Personalization (Generic vs Personalized), "
            f"pooling Task and Pipeline: {_latex_scientific(tc['PR(>F)'])} "
            "(preferred to naive Tukey on pooled rows; see caution below)."
        )
    tk = results["l2"]["tukey_condition"]
    if tk and "pvalue_generic_vs_personalized" in tk:
        pv = tk["pvalue_generic_vs_personalized"]
        lines.append(
            f"Naive Tukey HSD on condition labels alone (ignores Task/Pipeline/User): $p = {pv:.4g}$"
        )
    lines.append("")
    lines.append("--- IoU (higher often indicates more overlap) ---")
    lines.extend(
        prose_lines_metric(
            results["iou"]["anova_typ3"],
            results["iou"]["marginal_means"],
            int(results["iou"]["residual_df"]),
            "IoU",
            lower_is_better=False,
        )
    )
    lines.append("")
    lines.append("Full ANOVA tables (Type III) are in the JSON under anova_typ3 / anova_table_csv.")
    lines.append(
        "Caution: Tukey HSD assumes approximate independence; data are repeated measures "
        "within user. Prefer the model F-tests above for inference; Tukey is descriptive."
    )

    out_txt = Path(args.out_txt).expanduser()
    if not out_txt.is_absolute():
        out_txt = root / out_txt
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_txt}")
    print("\n".join(lines[:25]))


if __name__ == "__main__":
    main()
