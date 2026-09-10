#!/usr/bin/env python3
"""
Evaluate VAE-style personalized ProMPs on User 1 digit trajectories.

This mirrors the PCA-based evaluation in `personalize_digit_library_and_evaluate.py`,
but uses the ProMP-VAE style-personalized library:

- Generic ProMPs: per-digit ProMPs stored in `user1_digit_generic_library/task_<d>_initialpromp/promp.npz`
- Personalized ProMPs: per-task means and covariances from
  `User_1_ProMP_library_personalized_vae_style/promp_library_personalized_vae_style.npz`

For each digit task (0–9) and each demonstration:
- Reconstruct generic and personalized trajectories from their ProMP means
- Compute L2 distance and IoU between demo and each ProMP trajectory
- Aggregate mean and std of L2 and IoU per task

Outputs:
- Per-task bar plots (generic vs personalized) for L2 and IoU
- CSV files with per-task means/stds for L2 and IoU
- One "detailed" trajectory plot per digit (demo vs generic vs personalized)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Set matplotlib backend (headless-friendly)
import os

_root_dir = Path(__file__).resolve().parent
_mpl_cache = _root_dir / ".mplcache"
_cache_home = _root_dir / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_home))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist

np.seterr(all="ignore")

def mirror_then_rotate_180(traj: np.ndarray) -> np.ndarray:
    """
    Apply the requested visualization transform: mirror, then rotate 180°.

    In coordinate form, mirror across the y-axis (x -> -x), then rotate 180°
    ((x, y) -> (-x, -y)), which composes to (x, y) -> (x, -y).
    """
    out = traj.copy()
    out[:, 0] = -out[:, 0]  # mirror (x -> -x)
    out[:, 0] = -out[:, 0]  # rotate 180 (x -> -x) cancels mirror
    out[:, 1] = -out[:, 1]  # rotate 180 (y -> -y)
    return out


def read_trajectory_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read trajectory CSV file and return time and y coordinates."""
    data = np.genfromtxt(path, delimiter=",", skip_header=1).astype(float)
    if data.ndim == 1 or data.shape[1] < 3:
        raise ValueError(f"Expected CSV columns time,x,y in {path}")
    data = data[np.isfinite(data).all(axis=1)]
    time = data[:, 0]
    y = data[:, 1:3]
    return time, y


def normalize_time(time: np.ndarray) -> np.ndarray:
    """Normalize time to [0, 1] range."""
    t0 = float(time[0])
    t1 = float(time[-1])
    if t1 <= t0:
        return np.linspace(0.0, 1.0, num=time.shape[0])
    return (time - t0) / (t1 - t0)


def make_rbf_basis(t: np.ndarray, n_basis: int, width: float) -> np.ndarray:
    """Create RBF basis functions."""
    centers = np.linspace(0.0, 1.0, num=n_basis)
    diff = t[:, None] - centers[None, :]
    basis = np.exp(-0.5 * (diff / width) ** 2)
    basis /= basis.sum(axis=1, keepdims=True)
    return basis


def reconstruct_trajectory(
    weights: np.ndarray,
    n_basis: int = 20,
    width: float = 0.05,
    duration: float = 1.0,
    dt: float = 0.01,
) -> np.ndarray:
    """Reconstruct trajectory from ProMP weights."""
    d = 2
    weights_2d = weights.reshape(d, n_basis)
    t = np.arange(0, duration, dt)
    t_norm = t / duration
    centers = np.linspace(0.0, 1.0, num=n_basis)
    diff = t_norm[:, None] - centers[None, :]
    basis = np.exp(-0.5 * (diff / width) ** 2)
    basis /= basis.sum(axis=1, keepdims=True)
    trajectory = basis @ weights_2d.T
    return trajectory


def interpolate_trajectory(traj: np.ndarray, target_length: int) -> np.ndarray:
    """Interpolate trajectory to target length."""
    if len(traj) == target_length:
        return traj
    original_indices = np.linspace(0, len(traj) - 1, len(traj))
    target_indices = np.linspace(0, len(traj) - 1, target_length)
    interp_func = interp1d(
        original_indices,
        traj,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return interp_func(target_indices)


def calculate_l2_norm(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """Calculate L2-norm between two trajectories."""
    max_length = max(len(traj1), len(traj2))
    traj1_interp = interpolate_trajectory(traj1, max_length)
    traj2_interp = interpolate_trajectory(traj2, max_length)
    diff = traj1_interp - traj2_interp
    return float(np.sqrt(np.sum(diff**2)))


def calculate_trajectory_iou(
    traj1: np.ndarray,
    traj2: np.ndarray,
    threshold: float = 5.0,
) -> float:
    """Calculate IoU between two trajectories."""
    max_length = max(len(traj1), len(traj2))
    traj1_interp = interpolate_trajectory(traj1, max_length)
    traj2_interp = interpolate_trajectory(traj2, max_length)
    distances = cdist(traj1_interp, traj2_interp)
    min_distances_traj1 = np.min(distances, axis=1)
    matches_traj1 = (min_distances_traj1 <= threshold).sum()
    min_distances_traj2 = np.min(distances, axis=0)
    matches_traj2 = (min_distances_traj2 <= threshold).sum()
    intersection = (matches_traj1 + matches_traj2) / 2.0
    union = max_length
    if union == 0:
        return 0.0
    return float(intersection / union)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate VAE-style personalized digit ProMP library on User 1.",
    )
    parser.add_argument(
        "--digit_library_generic",
        type=str,
        default="user1_digit_generic_library",
        help="Path to digit generic library directory (with task_<d>_initialpromp).",
    )
    parser.add_argument(
        "--digit_trajectories",
        type=str,
        default="user_1_digit_library_output",
        help="Directory containing User 1 digit trajectory CSVs.",
    )
    parser.add_argument(
        "--style_stats_npz",
        type=str,
        default="ProMP/digit_style_stats_user1.npz",
        help="NPZ containing style_mu and style_cov computed in digit domain.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="ProMP/vis_vae_style_digit",
        help="Output directory for plots and CSV statistics.",
    )
    parser.add_argument("--n_basis", type=int, default=20, help="RBF basis count.")
    parser.add_argument("--width", type=float, default=0.05, help="RBF width.")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step.")
    parser.add_argument(
        "--style_strength",
        type=float,
        default=1.0,
        help=(
            "Strength of style integration in the Gaussian product. "
            "Implemented as a scale on style precision (inverse covariance): "
            "Σ_post^{-1} = Σ_task^{-1} + style_strength * Σ_style^{-1}. "
            "Use <1.0 to make style effect weaker (e.g., 0.1), >1.0 stronger."
        ),
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=5.0,
        help="Distance threshold for IoU calculation.",
    )
    args = parser.parse_args()

    digit_library_dir = Path(args.digit_library_generic)
    digit_trajectories_dir = Path(args.digit_trajectories)
    output_dir = Path(args.out_dir)

    print("=" * 70)
    print("EVALUATING DIGIT LIBRARY PERSONALIZATION (VAE STYLE INTEGRATION)")
    print("=" * 70)

    # Load digit-domain style statistics (style_mu, style_cov)
    print("\nLoading digit-domain style statistics...")
    style_data = np.load(Path(args.style_stats_npz), allow_pickle=True)
    style_mu = style_data["style_mu"]
    style_cov = style_data["style_cov"]

    # Load generic means and covariances from per-digit library
    print("Loading generic digit ProMPs...")
    generic_means: Dict[str, np.ndarray] = {}
    generic_covs: Dict[str, np.ndarray] = {}

    for task_id in range(10):
        task_str = str(task_id)
        promp_file = digit_library_dir / f"task_{task_str}_initialpromp" / "promp.npz"
        if not promp_file.exists():
            print(f"  Warning: generic ProMP missing for digit {task_str}: {promp_file}")
            continue

        promp_data = np.load(promp_file)
        generic_means[task_str] = promp_data["mean"]
        generic_covs[task_str] = promp_data["cov"]
        print(f"  Task {task_str}: loaded generic mean and covariance")

    # Evaluate all demos per digit
    print("\nEvaluating all digit demos...")
    output_dir.mkdir(parents=True, exist_ok=True)

    task_stats: Dict[str, Dict[str, float]] = {}

    for task_id in range(10):
        task_str = str(task_id)
        if task_str not in generic_means:
            continue

        task_folder = digit_trajectories_dir / f"task_{task_str}"
        if not task_folder.exists():
            print(f"  Task {task_str}: no trajectory folder found at {task_folder}")
            continue

        task_demos: List[Tuple[np.ndarray, np.ndarray, Path]] = []
        for csv_path in sorted(task_folder.rglob("*-trajectory.csv")):
            try:
                time, y = read_trajectory_csv(csv_path)
                if time.size < 2:
                    continue
                task_demos.append((time, y, csv_path))
            except Exception:
                continue

        if not task_demos:
            print(f"  Task {task_str}: no valid demos found")
            continue

        print(f"  Task {task_str}: {len(task_demos)} demos")

        l2_generic_list: List[float] = []
        l2_personalized_list: List[float] = []
        iou_generic_list: List[float] = []
        iou_personalized_list: List[float] = []

        generic_mean = generic_means[task_str]
        generic_cov = generic_covs[task_str]

        # Use first demo for a detailed trajectory plot
        detailed_demo_plotted = False

        for demo_idx, (time, y, csv_path) in enumerate(task_demos):
            # Combine generic task ProMP with style ProMP via Gaussian product:
            # Σ_personal^{-1} = Σ_task^{-1} + Σ_style^{-1}
            # μ_personal = Σ_personal ( Σ_task^{-1} μ_task + Σ_style^{-1} μ_style )
            eye = np.eye(generic_cov.shape[0])
            cov_task = generic_cov + 1e-6 * eye
            cov_style = style_cov + 1e-6 * eye
            cov_task_inv = np.linalg.inv(cov_task)
            cov_style_inv = np.linalg.inv(cov_style)
            cov_post_inv = cov_task_inv + float(args.style_strength) * cov_style_inv
            cov_post = np.linalg.inv(cov_post_inv)
            personalized_mean = cov_post @ (
                cov_task_inv @ generic_mean
                + float(args.style_strength) * (cov_style_inv @ style_mu)
            )

            duration = len(y) * args.dt
            generic_traj = reconstruct_trajectory(
                generic_mean,
                args.n_basis,
                args.width,
                duration,
                args.dt,
            )
            personalized_traj = reconstruct_trajectory(
                personalized_mean,
                args.n_basis,
                args.width,
                duration,
                args.dt,
            )

            if len(generic_traj) != len(y):
                generic_traj = interpolate_trajectory(generic_traj, len(y))
            if len(personalized_traj) != len(y):
                personalized_traj = interpolate_trajectory(personalized_traj, len(y))

            l2_gen = calculate_l2_norm(y, generic_traj)
            l2_pers = calculate_l2_norm(y, personalized_traj)
            iou_gen = calculate_trajectory_iou(y, generic_traj, args.iou_threshold)
            iou_pers = calculate_trajectory_iou(y, personalized_traj, args.iou_threshold)

            l2_generic_list.append(l2_gen)
            l2_personalized_list.append(l2_pers)
            iou_generic_list.append(iou_gen)
            iou_personalized_list.append(iou_pers)

            # Detailed trajectory plot for the first demo of this digit
            if not detailed_demo_plotted:
                y_plot = mirror_then_rotate_180(y)
                generic_plot = mirror_then_rotate_180(generic_traj)
                personalized_plot = mirror_then_rotate_180(personalized_traj)
                # Plot A: Demo vs Personalized only (requested)
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.plot(
                    y_plot[:, 0],
                    y_plot[:, 1],
                    "k-",
                    label="Demo",
                    linewidth=2,
                )
                ax.plot(
                    personalized_plot[:, 0],
                    personalized_plot[:, 1],
                    "r-",
                    label="Personalized ProMP (VAE style)",
                    linewidth=2,
                )
                ax.set_aspect("equal", "box")
                ax.set_title(f"Digit {task_str} – Demo vs Personalized (VAE style)")
                ax.legend()
                ax.grid(True, alpha=0.3, linestyle="--")
                detailed_file = (
                    output_dir / f"digit_task_{task_str}_detailed_vae_style.png"
                )
                plt.tight_layout()
                plt.savefig(detailed_file, dpi=200, bbox_inches="tight")
                plt.close(fig)

                # Plot B: Demo vs Generic vs Personalized (additional requested)
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                ax2.plot(
                    y_plot[:, 0],
                    y_plot[:, 1],
                    "k-",
                    label="Demo",
                    linewidth=2,
                )
                ax2.plot(
                    generic_plot[:, 0],
                    generic_plot[:, 1],
                    "b--",
                    label="PCA-Style",
                    linewidth=2,
                )
                ax2.plot(
                    personalized_plot[:, 0],
                    personalized_plot[:, 1],
                    "r-",
                    label="VAE style",
                    linewidth=2,
                )
                ax2.set_aspect("equal", "box")
                ax2.set_title(f"Digit {task_str} – Demo vs PCA-style vs VAE-style Personalized")
                ax2.legend()
                ax2.grid(True, alpha=0.3, linestyle="--")
                detailed_file2 = (
                    output_dir / f"digit_task_{task_str}_detailed_generic_vs_vae.png"
                )
                plt.tight_layout()
                plt.savefig(detailed_file2, dpi=200, bbox_inches="tight")
                plt.close(fig2)
                detailed_demo_plotted = True

        task_stats[task_str] = {
            "n_demos": len(task_demos),
            "l2_generic_mean": float(np.mean(l2_generic_list)),
            "l2_generic_std": float(np.std(l2_generic_list)),
            "l2_personalized_mean": float(np.mean(l2_personalized_list)),
            "l2_personalized_std": float(np.std(l2_personalized_list)),
            "iou_generic_mean": float(np.mean(iou_generic_list)),
            "iou_generic_std": float(np.std(iou_generic_list)),
            "iou_personalized_mean": float(np.mean(iou_personalized_list)),
            "iou_personalized_std": float(np.std(iou_personalized_list)),
        }

    # Aggregate plots
    tasks = sorted(task_stats.keys(), key=int)
    if not tasks:
        print("No tasks with statistics; nothing to plot.")
        return

    task_labels = [f"Digit {t}" for t in tasks]

    # L2 comparison
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    l2_gen_means = [task_stats[t]["l2_generic_mean"] for t in tasks]
    l2_gen_stds = [task_stats[t]["l2_generic_std"] for t in tasks]
    l2_pers_means = [task_stats[t]["l2_personalized_mean"] for t in tasks]
    l2_pers_stds = [task_stats[t]["l2_personalized_std"] for t in tasks]

    x_pos = np.arange(len(tasks))
    width_bar = 0.35

    ax1.bar(
        x_pos - width_bar / 2,
        l2_gen_means,
        width_bar,
        yerr=l2_gen_stds,
        label="Generic ProMP",
        alpha=0.8,
        color="blue",
        edgecolor="black",
        linewidth=1,
        capsize=5,
    )
    ax1.bar(
        x_pos + width_bar / 2,
        l2_pers_means,
        width_bar,
        yerr=l2_pers_stds,
        label="Personalized ProMP (VAE style)",
        alpha=0.8,
        color="red",
        edgecolor="black",
        linewidth=1,
        capsize=5,
    )

    ax1.set_xlabel("Digit", fontsize=14, fontweight="bold")
    ax1.set_ylabel("L2-norm (Mean ± Std)", fontsize=14, fontweight="bold")
    ax1.set_title(
        "L2-norm Comparison: Generic vs VAE-Style Personalized ProMP – Digit Library",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(task_labels, fontsize=11)
    ax1.legend(fontsize=12, loc="upper left")
    ax1.grid(True, alpha=0.3, axis="y", linestyle="--")

    plt.tight_layout()
    l2_plot_file = output_dir / "digit_library_l2_norm_comparison_vae_style.png"
    plt.savefig(l2_plot_file, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"L2-norm plot saved to: {l2_plot_file}")

    # IoU comparison
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    iou_gen_means = [task_stats[t]["iou_generic_mean"] for t in tasks]
    iou_gen_stds = [task_stats[t]["iou_generic_std"] for t in tasks]
    iou_pers_means = [task_stats[t]["iou_personalized_mean"] for t in tasks]
    iou_pers_stds = [task_stats[t]["iou_personalized_std"] for t in tasks]

    ax2.bar(
        x_pos - width_bar / 2,
        iou_gen_means,
        width_bar,
        yerr=iou_gen_stds,
        label="Generic ProMP",
        alpha=0.8,
        color="blue",
        edgecolor="black",
        linewidth=1,
        capsize=5,
    )
    ax2.bar(
        x_pos + width_bar / 2,
        iou_pers_means,
        width_bar,
        yerr=iou_pers_stds,
        label="Personalized ProMP (VAE style)",
        alpha=0.8,
        color="red",
        edgecolor="black",
        linewidth=1,
        capsize=5,
    )

    ax2.set_xlabel("Digit", fontsize=14, fontweight="bold")
    ax2.set_ylabel("IoU (Mean ± Std)", fontsize=14, fontweight="bold")
    ax2.set_title(
        f"IoU Comparison: Generic vs VAE-Style Personalized ProMP – Digit Library\n"
        f"(Threshold: {args.iou_threshold})",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(task_labels, fontsize=11)
    ax2.set_ylim([0, 1.1])
    ax2.legend(fontsize=12, loc="upper left")
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--")

    plt.tight_layout()
    iou_plot_file = output_dir / "digit_library_iou_comparison_vae_style.png"
    plt.savefig(iou_plot_file, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"IoU plot saved to: {iou_plot_file}")

    # CSV statistics: L2
    l2_csv_file = output_dir / "digit_library_l2_norm_statistics_vae_style.csv"
    with l2_csv_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Digit",
                "Generic_Mean",
                "Generic_Std",
                "Personalized_Mean",
                "Personalized_Std",
                "N_Demos",
            ],
        )
        for t in tasks:
            stats = task_stats[t]
            writer.writerow(
                [
                    t,
                    f"{stats['l2_generic_mean']:.4f}",
                    f"{stats['l2_generic_std']:.4f}",
                    f"{stats['l2_personalized_mean']:.4f}",
                    f"{stats['l2_personalized_std']:.4f}",
                    stats["n_demos"],
                ],
            )
    print(f"L2-norm statistics saved to: {l2_csv_file}")

    # CSV statistics: IoU
    iou_csv_file = output_dir / "digit_library_iou_statistics_vae_style.csv"
    with iou_csv_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Digit",
                "Generic_Mean",
                "Generic_Std",
                "Personalized_Mean",
                "Personalized_Std",
                "N_Demos",
            ],
        )
        for t in tasks:
            stats = task_stats[t]
            writer.writerow(
                [
                    t,
                    f"{stats['iou_generic_mean']:.4f}",
                    f"{stats['iou_generic_std']:.4f}",
                    f"{stats['iou_personalized_mean']:.4f}",
                    f"{stats['iou_personalized_std']:.4f}",
                    stats["n_demos"],
                ],
            )
    print(f"IoU statistics saved to: {iou_csv_file}")

    print("\n" + "=" * 70)
    print("DIGIT LIBRARY VAE-STYLE PERSONALIZATION EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

