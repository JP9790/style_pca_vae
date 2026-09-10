#!/usr/bin/env python3
"""
Create a detailed single plot for task b showing:
- Original trajectory from output folder
- Initial learned ProMP
- Personalized ProMP
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
    """
    Reconstruct trajectory from ProMP weights.
    
    Args:
        weights: Weight vector of shape (d * n_basis,)
        n_basis: Number of RBF basis functions
        width: RBF width
        duration: Trajectory duration in seconds
        dt: Time step
        
    Returns:
        Trajectory array of shape (T, 2) with x, y coordinates
    """
    # Reshape weights
    d = 2  # x, y dimensions
    weights_2d = weights.reshape(d, n_basis)
    
    # Generate time points
    t = np.arange(0, duration, dt)
    t_norm = t / duration  # Normalize to [0, 1]
    
    # Create RBF basis
    centers = np.linspace(0.0, 1.0, num=n_basis)
    diff = t_norm[:, None] - centers[None, :]
    basis = np.exp(-0.5 * (diff / width) ** 2)
    basis /= basis.sum(axis=1, keepdims=True)
    
    # Reconstruct trajectory
    trajectory = basis @ weights_2d.T  # (T, d)
    
    return trajectory


def interpolate_trajectory(traj: np.ndarray, target_length: int) -> np.ndarray:
    """
    Interpolate trajectory to target length using linear interpolation.
    
    Args:
        traj: Trajectory array of shape (T, 2)
        target_length: Target number of points
        
    Returns:
        Interpolated trajectory of shape (target_length, 2)
    """
    if len(traj) == target_length:
        return traj
    
    original_indices = np.linspace(0, len(traj) - 1, len(traj))
    target_indices = np.linspace(0, len(traj) - 1, target_length)
    
    interp_func = interp1d(original_indices, traj, axis=0, kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
    interpolated = interp_func(target_indices)
    
    return interpolated


def plot_task_b_detailed(
    original_library_file: Path,
    personalized_library_file: Path,
    trajectories_dir: Path,
    target_user: str,
    output_file: Path,
    n_basis: int = 20,
    width: float = 0.05
):
    """Create detailed single plot for task b."""
    # Load libraries
    orig_lib = np.load(original_library_file)
    pers_lib = np.load(personalized_library_file)
    
    task = 'b'
    
    # Load original trajectory from output folder
    original_traj = None
    for csv_path in trajectories_dir.rglob("*-trajectory.csv"):
        user_id = infer_user_id_from_path(csv_path)
        task_id = infer_task_id_from_path(csv_path)
        if user_id == target_user and task_id == task:
            try:
                time, y = read_trajectory_csv(csv_path)
                if time.size >= 2:
                    original_traj = y  # Store x, y coordinates
                    break
            except Exception as e:
                print(f"Warning: Could not load {csv_path}: {e}")
    
    if original_traj is None:
        raise ValueError(f"Could not find original trajectory for task {task} and user {target_user}")
    
    # Get ProMP means
    orig_mean = orig_lib[f'task_{task}_mean']
    pers_mean = pers_lib[f'task_{task}_mean']
    
    # Reconstruct ProMP trajectories
    duration = len(original_traj) * 0.01  # Assume 0.01s time step
    dt = 0.01
    
    orig_promp_traj = reconstruct_trajectory(orig_mean, n_basis, width, duration, dt)
    pers_promp_traj = reconstruct_trajectory(pers_mean, n_basis, width, duration, dt)
    
    # Interpolate ProMP trajectories to match original length
    if len(orig_promp_traj) != len(original_traj):
        orig_promp_traj = interpolate_trajectory(orig_promp_traj, len(original_traj))
    if len(pers_promp_traj) != len(original_traj):
        pers_promp_traj = interpolate_trajectory(pers_promp_traj, len(original_traj))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot original input trajectory (thick black line)
    ax.plot(original_traj[:, 0], original_traj[:, 1], 
           'k-', linewidth=4, label='Original Input', alpha=0.9, zorder=3)
    
    # Plot initial ProMP (blue solid line)
    ax.plot(orig_promp_traj[:, 0], orig_promp_traj[:, 1], 
           'b-', linewidth=3, label='Initial ProMP', alpha=0.8, zorder=2)
    
    # Plot personalized ProMP (red dashed line)
    ax.plot(pers_promp_traj[:, 0], pers_promp_traj[:, 1], 
           'r--', linewidth=3, label='Personalized ProMP', alpha=0.8, zorder=2)
    
    # Mark start point
    ax.plot(original_traj[0, 0], original_traj[0, 1], 
           'go', markersize=15, label='Start', zorder=5, 
           markeredgecolor='black', markeredgewidth=2)
    
    # Mark end points
    ax.plot(original_traj[-1, 0], original_traj[-1, 1], 
           'ks', markersize=15, label='End (Input)', zorder=5, 
           markeredgecolor='white', markeredgewidth=2)
    ax.plot(orig_promp_traj[-1, 0], orig_promp_traj[-1, 1], 
           'bs', markersize=12, label='End (Initial ProMP)', zorder=4, 
           markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(pers_promp_traj[-1, 0], pers_promp_traj[-1, 1], 
           'rs', markersize=12, label='End (Personalized ProMP)', zorder=4, 
           markeredgecolor='white', markeredgewidth=1.5)
    
    # Calculate L2-norms for display
    diff_orig = original_traj - orig_promp_traj
    diff_pers = original_traj - pers_promp_traj
    l2_orig = np.sqrt(np.sum(diff_orig ** 2))
    l2_pers = np.sqrt(np.sum(diff_pers ** 2))
    
    ax.set_xlabel('X Position', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Position', fontsize=14, fontweight='bold')
    ax.set_title(f'Detailed Trajectory Comparison: Task {task.upper()} (User {target_user})\n'
                f'L2-norm: Initial ProMP={l2_orig:.2f}, Personalized ProMP={l2_pers:.2f}', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed trajectory plot saved to: {output_file}")
    print(f"L2-norm (Original vs Initial ProMP): {l2_orig:.4f}")
    print(f"L2-norm (Original vs Personalized ProMP): {l2_pers:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot detailed trajectory comparison for task b.")
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
        help="Path to personalized library (default: User_{target_user}_ProMP_library_personalized/promp_library_personalized.npz).",
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
        help="Output visualization path (default: ProMP/vis/task_b_detailed_user_{target_user}.png).",
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
        output_file = Path(f"ProMP/vis/task_b_detailed_user_{args.target_user}.png")
    else:
        output_file = Path(args.out)
    
    trajectories_dir = Path(args.trajectories_dir)
    
    if not original_library_file.exists():
        raise FileNotFoundError(f"Original library not found: {original_library_file}")
    if not personalized_library_file.exists():
        raise FileNotFoundError(f"Personalized library not found: {personalized_library_file}")
    
    plot_task_b_detailed(
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
