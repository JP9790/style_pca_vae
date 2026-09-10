#!/usr/bin/env python3
"""
Create a single bar plot comparing L2-norms between original trajectories and ProMPs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

# Set matplotlib backend
import os
_root_dir = Path(__file__).resolve().parent
_mpl_cache = _root_dir / ".mplcache"
_cache_home = _root_dir / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_home))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def read_trajectory_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read trajectory CSV file and return time and y coordinates."""
    data = np.genfromtxt(path, delimiter=",", skip_header=1).astype(float)
    if data.ndim == 1 or data.shape[1] < 3:
        raise ValueError(f"Expected CSV columns time,x,y in {path}")
    data = data[np.isfinite(data).all(axis=1)]
    time = data[:, 0]
    y = data[:, 1:3]
    return time, y


def infer_task_id_from_path(csv_path: Path) -> str:
    """Extract task ID from folder structure (e.g., task_a -> a)."""
    parts = csv_path.parts
    for part in parts:
        if part.startswith("task_"):
            return part.replace("task_", "")
    return "unknown"


def infer_user_id_from_path(csv_path: Path) -> str:
    """Extract user ID from folder structure (e.g., User_1 -> 1)."""
    parts = csv_path.parts
    for part in parts:
        if part.startswith("User_"):
            return part.replace("User_", "")
    return "unknown"


def reconstruct_trajectory(weights: np.ndarray, n_basis: int = 20, width: float = 0.05, 
                          duration: float = 1.0, dt: float = 0.01) -> np.ndarray:
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
    interp_func = interp1d(original_indices, traj, axis=0, kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
    return interp_func(target_indices)


def calculate_l2_norm(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """Calculate L2-norm between two trajectories."""
    max_length = max(len(traj1), len(traj2))
    traj1_interp = interpolate_trajectory(traj1, max_length)
    traj2_interp = interpolate_trajectory(traj2, max_length)
    diff = traj1_interp - traj2_interp
    return float(np.sqrt(np.sum(diff ** 2)))


def plot_l2_norm_comparison(
    original_library_file: Path,
    personalized_library_file: Path,
    trajectories_dir: Path,
    target_user: str,
    output_file: Path,
    n_basis: int = 20,
    width: float = 0.05
):
    """Create L2-norm comparison bar plot."""
    # Load libraries
    orig_lib = np.load(original_library_file)
    pers_lib = np.load(personalized_library_file)
    orig_tasks = orig_lib['tasks'].astype(str)
    
    # Load original trajectories
    print(f"Loading original trajectories for user {target_user}...")
    original_trajectories = {}
    for csv_path in trajectories_dir.rglob("*-trajectory.csv"):
        user_id = infer_user_id_from_path(csv_path)
        if user_id == target_user:
            try:
                time, y = read_trajectory_csv(csv_path)
                if time.size < 2:
                    continue
                task_id = infer_task_id_from_path(csv_path)
                if task_id in orig_tasks:
                    original_trajectories[task_id] = y
            except Exception as e:
                print(f"Warning: Skipping {csv_path}: {e}")
    
    print(f"Loaded {len(original_trajectories)} original trajectories")
    
    # Calculate L2-norms
    print("Calculating L2-norms...")
    l2_norms_original = []
    l2_norms_personalized = []
    task_labels = []
    
    for task in sorted(orig_tasks):
        if task not in original_trajectories:
            continue
        
        task_labels.append(task)
        original_traj = original_trajectories[task]
        
        orig_mean = orig_lib[f'task_{task}_mean']
        pers_mean = pers_lib[f'task_{task}_mean']
        
        duration = len(original_traj) * 0.01
        dt = 0.01
        
        orig_promp_traj = reconstruct_trajectory(orig_mean, n_basis, width, duration, dt)
        pers_promp_traj = reconstruct_trajectory(pers_mean, n_basis, width, duration, dt)
        
        if len(orig_promp_traj) != len(original_traj):
            orig_promp_traj = interpolate_trajectory(orig_promp_traj, len(original_traj))
        if len(pers_promp_traj) != len(original_traj):
            pers_promp_traj = interpolate_trajectory(pers_promp_traj, len(original_traj))
        
        l2_orig = calculate_l2_norm(original_traj, orig_promp_traj)
        l2_pers = calculate_l2_norm(original_traj, pers_promp_traj)
        
        l2_norms_original.append(l2_orig)
        l2_norms_personalized.append(l2_pers)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(task_labels))
    width_bar = 0.35
    
    bars1 = ax.bar(x_pos - width_bar/2, l2_norms_original, width_bar, 
                   label='Original Input vs Initial ProMP', alpha=0.8, 
                   color='blue', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x_pos + width_bar/2, l2_norms_personalized, width_bar, 
                   label='Original Input vs Personalized ProMP', alpha=0.8, 
                   color='red', edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Task', fontsize=14, fontweight='bold')
    ax.set_ylabel('L2-norm', fontsize=14, fontweight='bold')
    ax.set_title(f'L2-norm Comparison: Original Trajectories vs ProMP (User {target_user})', 
                fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(task_labels, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add statistics text
    avg_orig = np.mean(l2_norms_original)
    avg_pers = np.mean(l2_norms_personalized)
    improvement = ((avg_orig - avg_pers) / avg_orig * 100) if avg_orig > 0 else 0
    
    stats_text = f'Average L2-norm:\nInitial: {avg_orig:.2f}\nPersonalized: {avg_pers:.2f}\n'
    if improvement > 0:
        stats_text += f'Improvement: {improvement:.1f}%'
    else:
        stats_text += f'Change: {improvement:.1f}%'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"L2-norm comparison plot saved to: {output_file}")
    print(f"Average L2-norm (Initial): {avg_orig:.4f}")
    print(f"Average L2-norm (Personalized): {avg_pers:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot L2-norm comparison bar chart.")
    parser.add_argument(
        "--original_library",
        type=str,
        default="User_1_ProMP_library/promp_library.npz",
        help="Path to original ProMP library.",
    )
    parser.add_argument(
        "--personalized_library",
        type=str,
        default=None,
        help="Path to personalized library.",
    )
    parser.add_argument(
        "--target_user",
        type=str,
        default="1",
        help="Target user ID.",
    )
    parser.add_argument(
        "--trajectories_dir",
        type=str,
        default="output",
        help="Directory containing original trajectory CSVs.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output visualization path.",
    )
    parser.add_argument("--n_basis", type=int, default=20, help="RBF basis count.")
    parser.add_argument("--width", type=float, default=0.05, help="RBF width.")
    args = parser.parse_args()
    
    original_library_file = Path(args.original_library)
    
    if args.personalized_library is None:
        personalized_library_file = Path(f"User_{args.target_user}_ProMP_library_personalized/promp_library_personalized.npz")
    else:
        personalized_library_file = Path(args.personalized_library)
    
    if args.out is None:
        output_file = Path(f"ProMP/vis/l2_norm_comparison_user_{args.target_user}.png")
    else:
        output_file = Path(args.out)
    
    trajectories_dir = Path(args.trajectories_dir)
    
    if not original_library_file.exists():
        raise FileNotFoundError(f"Original library not found: {original_library_file}")
    if not personalized_library_file.exists():
        raise FileNotFoundError(f"Personalized library not found: {personalized_library_file}")
    
    plot_l2_norm_comparison(
        original_library_file,
        personalized_library_file,
        trajectories_dir,
        args.target_user,
        output_file,
        args.n_basis,
        args.width
    )


if __name__ == "__main__":
    main()
