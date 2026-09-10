#!/usr/bin/env python3
"""
Partition digit-domain VAE latent dimensions into task-related vs style-related.

For a single user (User 1 digits), we cannot do two-factor ANOVA with user.
So we follow the per-user procedure described earlier:
  - For each latent dim j, run one-way ANOVA across digit tasks (0..9).
    If p_task < alpha => task-related.
    Else => candidate style.
  - For candidate style dims, run one-sample t-test vs 0 across all demos.
    If t_p < alpha => style, else => noise.

Input:
  ProMP/promp_vae_digit/promp_vae_digit_user_1_latents.npz

Output:
  ProMP/vae_style_anova_digit_results.json
  ProMP/vae_style_anova_digit_results.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from scipy import stats
except Exception as exc:  # pragma: no cover
    raise SystemExit("scipy is required for ANOVA and t-tests") from exc

np.seterr(all="ignore")


def one_way_anova(y: np.ndarray, tasks: np.ndarray) -> float:
    """Return p-value for one-way ANOVA across task groups."""
    groups: List[np.ndarray] = []
    for t in sorted(set(tasks.tolist())):
        grp = y[tasks == t]
        if grp.size > 0:
            groups.append(grp)
    if len(groups) <= 1:
        return 1.0
    f_stat, p = stats.f_oneway(*groups)
    return float(p) if np.isfinite(p) else 1.0


def main() -> None:
    p = argparse.ArgumentParser(description="One-way ANOVA partitioning for digit-domain VAE latents (User 1).")
    p.add_argument(
        "--latents",
        type=str,
        default="ProMP/promp_vae_digit/promp_vae_digit_user_1_latents.npz",
        help="Digit VAE latents NPZ.",
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out_json", type=str, default="ProMP/vae_style_anova_digit_results.json")
    p.add_argument("--out_csv", type=str, default="ProMP/vae_style_anova_digit_results.csv")
    args = p.parse_args()

    data = np.load(Path(args.latents), allow_pickle=True)
    mu = data["mu"]  # (N, M)
    tasks = data["tasks"].astype(str)

    n, m = mu.shape
    results: List[Dict] = []

    for j in range(m):
        y = mu[:, j]
        p_task = one_way_anova(y, tasks)
        mean = float(np.mean(y))
        t_stat, t_p = stats.ttest_1samp(y, 0.0)
        t_p = float(t_p) if np.isfinite(t_p) else 1.0
        t_stat = float(t_stat) if np.isfinite(t_stat) else 0.0

        if p_task < args.alpha:
            label = "task"
        else:
            # candidate style -> refine with t-test vs 0
            label = "style" if t_p < args.alpha else "noise"

        results.append(
            {
                "component": j + 1,
                "p_task": float(p_task),
                "t_p": t_p,
                "t_stat": t_stat,
                "mean": mean,
                "label": label,
            },
        )

    label_counts: Dict[str, int] = {}
    for r in results:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "alpha": args.alpha,
                "n_samples": int(n),
                "n_components": int(m),
                "task_levels": sorted(set(tasks.tolist())),
                "label_counts": label_counts,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("component,p_task,t_stat,t_p,mean,label\n")
        for r in results:
            f.write(
                f"{r['component']},{r['p_task']:.6f},{r['t_stat']:.6f},{r['t_p']:.6f},{r['mean']:.6f},{r['label']}\n"
            )

    print(f"Saved JSON: {out_json}")
    print(f"Saved CSV:  {out_csv}")
    print("Label counts:", label_counts)


if __name__ == "__main__":
    main()

