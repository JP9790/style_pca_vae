#!/usr/bin/env python3
"""
Partition PCA latent dimensions into task-related vs style-related variables
using two-factor ANOVA with task and user as factors.

Following the paper description:
- For each principal component j, perform two-factor ANOVA with task and user as factors
- If task is statistically significant (p < 0.05), component is task-related
- If user is statistically significant while task is not, component is a candidate style dimension
- Components with no significant dependence on either factor are treated as noise
- For candidate style dimensions, use one-sample t-test to check if mean differs from zero
- Style-related components should have non-zero mean (systematic deviation)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy import stats
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("scipy is required for ANOVA and t-tests") from exc

np.seterr(all="ignore")


def infer_user_id(path: str) -> str:
    """Extract user ID from path if not available in data."""
    parts = Path(path).parts
    for part in parts:
        if part.startswith("User_"):
            return part.replace("User_", "")
    for part in parts:
        if part.startswith("f") and "_" in part:
            return part
    return "unknown"


def one_hot(labels: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Create one-hot encoding matrix for categorical variables."""
    uniq = sorted(set(labels))
    if len(uniq) <= 1:
        return np.zeros((len(labels), 0)), uniq
    idx = {name: i for i, name in enumerate(uniq)}
    mat = np.zeros((len(labels), len(uniq) - 1))
    for r, name in enumerate(labels):
        col = idx[name] - 1
        if col >= 0:
            mat[r, col] = 1.0
    return mat, uniq


def fit_sse(y: np.ndarray, X: np.ndarray) -> Tuple[float, int]:
    """Fit linear model and return sum of squared errors and degrees of freedom."""
    if X.size == 0:
        y_hat = np.mean(y) * np.ones_like(y)
        resid = y - y_hat
        return float(resid @ resid), len(y) - 1
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid), len(y) - X.shape[1]


def f_test(
    sse_reduced: float, sse_full: float, df_reduced: int, df_full: int, df_num: int
) -> float:
    """
    Perform F-test to compare reduced model vs full model.
    Returns p-value for the F-statistic.
    """
    if df_num <= 0 or df_full <= 0:
        return 1.0
    denom = sse_full / max(df_full, 1)
    if denom <= 0:
        return 1.0
    f_stat = ((sse_reduced - sse_full) / df_num) / denom
    if not np.isfinite(f_stat) or f_stat < 0:
        return 1.0
    return float(1.0 - stats.f.cdf(f_stat, df_num, df_full))


def two_factor_anova(
    y: np.ndarray, tasks: np.ndarray, users: np.ndarray
) -> Tuple[float, float]:
    """
    Perform two-factor ANOVA with task and user as factors.
    
    This uses the general linear model approach:
    - Full model: y ~ intercept + task + user
    - Task effect: compare full model vs model without task (y ~ intercept + user)
    - User effect: compare full model vs model without user (y ~ intercept + task)
    
    Returns:
        p_task: p-value for task factor
        p_user: p-value for user factor
    """
    n = len(y)
    intercept = np.ones((n, 1))
    
    # Create design matrices
    task_mat, _ = one_hot(tasks.tolist())
    user_mat, _ = one_hot(users.tolist())
    
    # Full model: y ~ intercept + task + user
    X_full = np.hstack([intercept, task_mat, user_mat])
    sse_full, df_full = fit_sse(y, X_full)
    
    # Reduced model without task: y ~ intercept + user
    X_task_reduced = np.hstack([intercept, user_mat])
    sse_task_reduced, df_task_reduced = fit_sse(y, X_task_reduced)
    
    # Reduced model without user: y ~ intercept + task
    X_user_reduced = np.hstack([intercept, task_mat])
    sse_user_reduced, df_user_reduced = fit_sse(y, X_user_reduced)
    
    # Test task effect: compare full vs reduced (without task)
    df_task_num = df_task_reduced - df_full
    p_task = f_test(
        sse_task_reduced, sse_full, df_task_reduced, df_full, df_task_num
    )
    
    # Test user effect: compare full vs reduced (without user)
    df_user_num = df_user_reduced - df_full
    p_user = f_test(
        sse_user_reduced, sse_full, df_user_reduced, df_full, df_user_num
    )
    
    return p_task, p_user


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-factor ANOVA to partition PCA components into task-related vs style-related."
    )
    parser.add_argument(
        "--pca",
        type=str,
        default="ProMP/promp_pca_results.npz",
        help="Path to PCA results .npz.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for ANOVA and t-tests.",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="ProMP/style_anova_results.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="ProMP/style_anova_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Report top-K components by absolute mean as style candidates.",
    )
    args = parser.parse_args()

    # Load PCA results
    data = np.load(Path(args.pca))
    scores = data["scores"]  # (N, P) - latent coordinates z_i^(j)
    tasks = data["tasks"].astype(str)
    
    # Get user IDs - prefer from saved data, fall back to path extraction
    if "users" in data:
        users = data["users"].astype(str)
    else:
        paths = data["paths"].astype(str).tolist()
        users = np.array([infer_user_id(p) for p in paths])

    print(f"Analyzing {scores.shape[1]} principal components")
    print(f"Number of samples: {scores.shape[0]}")
    print(f"Number of tasks: {len(np.unique(tasks))}")
    print(f"Number of users: {len(np.unique(users))}")
    print(f"Significance level (alpha): {args.alpha}")
    print()

    # Get unique levels for reporting
    task_levels = sorted(set(tasks.tolist()))
    user_levels = sorted(set(users.tolist()))

    # Analyze each principal component
    results = []
    for j in range(scores.shape[1]):
        # Get latent scores for component j: z^(j)
        y = scores[:, j]
        
        # Perform two-factor ANOVA: test task and user effects
        p_task, p_user = two_factor_anova(y, tasks, users)
        
        # Compute mean for one-sample t-test
        mean = float(np.mean(y))
        
        # One-sample t-test: test if mean differs significantly from zero
        # This is used to evaluate candidate style dimensions
        t_stat, t_p = stats.ttest_1samp(y, 0.0)
        t_p = float(t_p) if np.isfinite(t_p) else 1.0
        
        # Label component according to paper description:
        # 1. If task is significant (p < alpha), component is task-related
        # 2. If user is significant while task is not, component is candidate style
        # 3. For candidate style, if t-test shows non-zero mean, label as style
        # 4. Otherwise, label as noise
        if p_task < args.alpha:
            label = "task"
        elif p_user < args.alpha and p_task >= args.alpha:
            # User is significant, task is not -> candidate style dimension
            if t_p < args.alpha:
                # Mean differs significantly from zero -> style-related
                label = "style"
            else:
                # Mean not significantly different from zero -> candidate but not confirmed
                label = "style_candidate"
        else:
            # Neither task nor user is significant -> noise/residual variability
            label = "noise"

        results.append(
            {
                "component": j + 1,
                "p_task": p_task,
                "p_user": p_user,
                "t_p": t_p,
                "t_stat": float(t_stat) if np.isfinite(t_stat) else 0.0,
                "mean": mean,
                "label": label,
            }
        )

    # Count components by label
    label_counts = {}
    for r in results:
        label = r["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    print("Component classification summary:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} components")

    # Get top-K style candidates by absolute mean
    top_k = max(args.top_k, 0)
    top_k_list = []
    if top_k > 0:
        ranked = sorted(results, key=lambda r: abs(r["mean"]), reverse=True)
        top_k_list = [
            {
                "component": r["component"],
                "mean": r["mean"],
                "abs_mean": abs(r["mean"]),
                "label": r["label"],
                "p_task": r["p_task"],
                "p_user": r["p_user"],
                "t_p": r["t_p"],
            }
            for r in ranked[:top_k]
        ]

    # Save results to JSON
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "alpha": args.alpha,
                "n_components": len(results),
                "task_levels": task_levels,
                "user_levels": user_levels,
                "label_counts": label_counts,
                "top_k_by_abs_mean": top_k_list,
                "results": results,
            },
            f,
            indent=2,
        )

    # Save results to CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("component,p_task,p_user,t_stat,t_p,mean,label\n")
        for row in results:
            f.write(
                f"{row['component']},{row['p_task']:.6f},{row['p_user']:.6f},"
                f"{row['t_stat']:.6f},{row['t_p']:.6f},{row['mean']:.6f},{row['label']}\n"
            )

    print(f"\nResults saved to:")
    print(f"  JSON: {out_json}")
    print(f"  CSV: {out_csv}")
    
    if top_k_list:
        print(f"\nTop {top_k} components by absolute mean:")
        for item in top_k_list:
    print(
                f"  Component {item['component']}: mean={item['mean']:.4f}, "
                f"label={item['label']}, p_task={item['p_task']:.4f}, "
                f"p_user={item['p_user']:.4f}, t_p={item['t_p']:.4f}"
    )


if __name__ == "__main__":
    main()
