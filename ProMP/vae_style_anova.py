#!/usr/bin/env python3
"""
Partition ProMP-VAE latent dimensions into task-related vs style-related variables.

This implements the procedure described in the VAE section:

1. For each latent variable z_m (m = 1..M), analyze encoder means μ_{z_{i,m}}
   across trajectories i using ANOVA with task identity as grouping factor.
2. Latent variables with significant task dependence (p_task < alpha) are
   interpreted as task-related dimensions.
3. Latent variables without significant task dependence (p_task >= alpha) are
   considered candidate style-related dimensions.
4. For candidate style dimensions, apply a one-sample t-test to check whether
   the mean differs significantly from zero. Dimensions with means not
   significantly different from zero are treated as noise-related, while those
   with systematic non-zero means are labeled style-related.
5. To incorporate multiple users, we additionally include user identity as a
   factor and perform a two-factor ANOVA (task, user) across all users:
      - Task effect: variation across tasks (task-related)
      - User effect: variation across users (style-related)

Labeling rules (mirroring the PCA-based analysis):
    - If task is significant (p_task < alpha): label = "task"
    - Else if user is significant (p_user < alpha) and task is not:
        - If t-test indicates non-zero mean (t_p < alpha): label = "style"
        - Else: label = "style_candidate"
    - Else: label = "noise"

Input:
    One or more NPZ files created by `promp_vae.py`, typically named:
        ProMP/promp_vae/promp_vae_user_<id>_latents.npz
    Each file must contain:
        mu      : (N_u, M) latent means for user u
        tasks   : (N_u,)   task labels (strings)
        paths   : (N_u,)   original CSV paths (used to infer user IDs)

Output:
    JSON and CSV summaries with per-latent statistics and labels.
"""
from __future__ import annotations

import argparse
import glob
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
    """Extract user ID from a path such as .../User_1/task_..."""
    parts = Path(path).parts
    for part in parts:
        if part.startswith("User_"):
            return part.replace("User_", "")
    # Digit data often uses folder names like user_1_digit_library_output
    for part in parts:
        if part.startswith("user_") and "_digit_" in part:
            # user_1_digit_library_output -> 1
            try:
                return part.split("user_")[1].split("_")[0]
            except Exception:  # noqa: BLE001
                return "unknown"
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
    sse_reduced: float,
    sse_full: float,
    df_reduced: int,
    df_full: int,
    df_num: int,
) -> float:
    """F-test to compare reduced vs full model, returning p-value."""
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
    y: np.ndarray,
    tasks: np.ndarray,
    users: np.ndarray,
) -> Tuple[float, float]:
    """
    Two-factor ANOVA with task and user as factors (general linear model).

    Full model:       y ~ intercept + task + user
    Task effect:      compare full vs model without task   (intercept + user)
    User effect:      compare full vs model without user   (intercept + task)
    """
    n = len(y)
    intercept = np.ones((n, 1))

    task_mat, _ = one_hot(tasks.tolist())
    user_mat, _ = one_hot(users.tolist())

    X_full = np.hstack([intercept, task_mat, user_mat])
    sse_full, df_full = fit_sse(y, X_full)

    X_task_reduced = np.hstack([intercept, user_mat])
    sse_task_reduced, df_task_reduced = fit_sse(y, X_task_reduced)

    X_user_reduced = np.hstack([intercept, task_mat])
    sse_user_reduced, df_user_reduced = fit_sse(y, X_user_reduced)

    df_task_num = df_task_reduced - df_full
    p_task = f_test(
        sse_task_reduced,
        sse_full,
        df_task_reduced,
        df_full,
        df_task_num,
    )

    df_user_num = df_user_reduced - df_full
    p_user = f_test(
        sse_user_reduced,
        sse_full,
        df_user_reduced,
        df_full,
        df_user_num,
    )

    return p_task, p_user


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ANOVA-based partitioning of ProMP-VAE latent dimensions "
        "into task-related vs style-related variables.",
    )
    parser.add_argument(
        "--latents_dir",
        type=str,
        default="ProMP/promp_vae",
        help="Directory containing *_latents.npz files from promp_vae.py.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="promp_vae_user_*_latents.npz",
        help="Glob pattern (relative to latents_dir) for latent NPZ files.",
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
        default="ProMP/vae_style_anova_results.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="ProMP/vae_style_anova_results.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    latents_root = Path(args.latents_dir).expanduser().resolve()
    pattern_path = latents_root / args.pattern
    files = sorted(glob.glob(str(pattern_path)))

    if not files:
        raise SystemExit(f"No latent NPZ files found for pattern: {pattern_path}")

    print("Loading latent representations from:")
    for f in files:
        print(f"  {f}")

    all_mu = []
    all_tasks = []
    all_users = []

    for f in files:
        data = np.load(f, allow_pickle=True)
        if "mu" not in data or "tasks" not in data:
            print(f"Warning: skipping {f} (missing 'mu' or 'tasks')")
            continue

        mu = data["mu"]  # (N_u, M)
        tasks = data["tasks"].astype(str)  # (N_u,)

        if "paths" in data:
            paths = data["paths"].astype(str).tolist()
            users = np.array([infer_user_id(p) for p in paths])
        else:
            # Fallback: infer user from filename
            fname = Path(f).name
            uid = "unknown"
            if "user_" in fname:
                # e.g., promp_vae_user_1_latents.npz
                try:
                    uid = fname.split("user_")[1].split("_")[0]
                except Exception:  # noqa: BLE001
                    uid = "unknown"
            users = np.array([uid] * mu.shape[0])

        if mu.shape[0] != tasks.shape[0] or mu.shape[0] != users.shape[0]:
            print(f"Warning: mismatched lengths in {f}, skipping.")
            continue

        all_mu.append(mu)
        all_tasks.append(tasks)
        all_users.append(users)

    if not all_mu:
        raise SystemExit("No valid latent data loaded.")

    mu_all = np.vstack(all_mu)  # (N, M)
    tasks_all = np.concatenate(all_tasks)  # (N,)
    users_all = np.concatenate(all_users)  # (N,)

    n_samples, n_components = mu_all.shape

    print(f"\nTotal samples: {n_samples}")
    print(f"Latent dimensionality (M): {n_components}")
    print(f"Number of tasks: {len(np.unique(tasks_all))}")
    print(f"Number of users: {len(np.unique(users_all))}")
    print(f"Significance level (alpha): {args.alpha}\n")

    task_levels = sorted(set(tasks_all.tolist()))
    user_levels = sorted(set(users_all.tolist()))

    results: List[Dict] = []

    for j in range(n_components):
        y = mu_all[:, j]

        # Two-factor ANOVA: task and user
        p_task, p_user = two_factor_anova(y, tasks_all, users_all)

        # One-sample t-test vs zero (mean behavior)
        mean = float(np.mean(y))
        t_stat, t_p = stats.ttest_1samp(y, 0.0)
        t_p = float(t_p) if np.isfinite(t_p) else 1.0
        t_stat = float(t_stat) if np.isfinite(t_stat) else 0.0

        # Label according to described criteria
        if p_task < args.alpha:
            label = "task"
        elif p_user < args.alpha and p_task >= args.alpha:
            if t_p < args.alpha:
                label = "style"
            else:
                label = "style_candidate"
        else:
            label = "noise"

        results.append(
            {
                "component": j + 1,
                "p_task": float(p_task),
                "p_user": float(p_user),
                "t_p": t_p,
                "t_stat": t_stat,
                "mean": mean,
                "label": label,
            },
        )

    # Count by label
    label_counts: Dict[str, int] = {}
    for r in results:
        label = r["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    print("Component classification summary:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} components")

    # Save JSON
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
                "results": results,
            },
            f,
            indent=2,
        )

    # Save CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("component,p_task,p_user,t_stat,t_p,mean,label\n")
        for row in results:
            f.write(
                f"{row['component']},{row['p_task']:.6f},{row['p_user']:.6f},"
                f"{row['t_stat']:.6f},{row['t_p']:.6f},{row['mean']:.6f},{row['label']}\n",
            )

    print(f"\nResults saved to:")
    print(f"  JSON: {out_json}")
    print(f"  CSV:  {out_csv}")


if __name__ == "__main__":
    main()

