#!/usr/bin/env python3
"""
Plot detailed trajectories for tasks p, s, u showing:
- Original trajectory (User 1)
- Generic (initial) ProMP trajectory
- Personalized ProMP trajectory
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


def plot_detailed_tasks(
    tasks: list,
    generic_library_dir: Path,
    user1_personalized_lib_file: Path,
    user1_trajectories_dir: Path,
    output_file: Path,
    n_basis: int = 20,
    width: float = 0.05,
    dt: float = 0.01
):
    """Plot detailed trajectories for specified tasks."""
    # Load User 1's style offset
    user1_pers_lib = np.load(user1_personalized_lib_file)
    style_offset = user1_pers_lib['style_offset']
    
    # Load User 1's original trajectories
    user1_trajectories = {}
    for csv_path in user1_trajectories_dir.rglob("*-trajectory.csv"):
        user_id = infer_user_id_from_path(csv_path)
        if user_id == "1":
            try:
                time, y = read_trajectory_csv(csv_path)
                if time.size < 2:
                    continue
                task_id = infer_task_id_from_path(csv_path)
                user1_trajectories[task_id] = y
            except Exception as e:
                print(f"Warning: Skipping {csv_path}: {e}")
    
    # Create figure with subplots for each task
    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(6*n_tasks, 6))
    if n_tasks == 1:
        axes = [axes]
    
    for idx, task in enumerate(tasks):
        ax = axes[idx]
        
        # Check if User 1 has this task
        if task not in user1_trajectories:
            ax.text(0.5, 0.5, f'Task {task.upper()}\nUser 1 does not have this task', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Task {task.upper()}', fontsize=14, fontweight='bold')
            continue
        
        user1_traj = user1_trajectories[task]
        
        # Load generic ProMP
        promp_file = generic_library_dir / f"task_{task}_initialpromp" / "promp.npz"
        if not promp_file.exists():
            ax.text(0.5, 0.5, f'Task {task.upper()}\nGeneric ProMP not found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'Task {task.upper()}', fontsize=14, fontweight='bold')
            continue
        
        initial_promp = np.load(promp_file)
        initial_mean = initial_promp['mean']
        personalized_mean = initial_mean + style_offset
        
        # Reconstruct trajectories
        duration = len(user1_traj) * dt
        initial_traj = reconstruct_trajectory(initial_mean, n_basis, width, duration, dt)
        personalized_traj = reconstruct_trajectory(personalized_mean, n_basis, width, duration, dt)
        
        # Interpolate to match User 1 trajectory length
        if len(initial_traj) != len(user1_traj):
            initial_traj = interpolate_trajectory(initial_traj, len(user1_traj))
        if len(personalized_traj) != len(user1_traj):
            personalized_traj = interpolate_trajectory(personalized_traj, len(user1_traj))
        
        # Calculate metrics for display
        diff_init = user1_traj - initial_traj
        diff_pers = user1_traj - personalized_traj
        l2_init = np.sqrt(np.sum(diff_init ** 2))
        l2_pers = np.sqrt(np.sum(diff_pers ** 2))
        
        # Plot trajectories
        ax.plot(user1_traj[:, 0], user1_traj[:, 1], 
               'k-', linewidth=4, label='Original (User 1)', alpha=0.9, zorder=3)
        ax.plot(initial_traj[:, 0], initial_traj[:, 1], 
               'b-', linewidth=3, label='Generic (Initial) ProMP', alpha=0.8, zorder=2)
        ax.plot(personalized_traj[:, 0], personalized_traj[:, 1], 
               'r--', linewidth=3, label='Personalized ProMP', alpha=0.8, zorder=2)
        
        # Mark start point
        ax.plot(user1_traj[0, 0], user1_traj[0, 1], 
               'go', markersize=15, label='Start', zorder=5, 
               markeredgecolor='black', markeredgewidth=2)
        
        # Mark end points
        ax.plot(user1_traj[-1, 0], user1_traj[-1, 1], 
               'ks', markersize=15, label='End (Original)', zorder=5, 
               markeredgecolor='white', markeredgewidth=2)
        ax.plot(initial_traj[-1, 0], initial_traj[-1, 1], 
               'bs', markersize=12, label='End (Initial)', zorder=4, 
               markeredgecolor='white', markeredgewidth=1.5)
        ax.plot(personalized_traj[-1, 0], personalized_traj[-1, 1], 
               'rs', markersize=12, label='End (Personalized)', zorder=4, 
               markeredgecolor='white', markeredgewidth=1.5)
        
        ax.set_xlabel('X Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=12, fontweight='bold')
        ax.set_title(f'Task {task.upper()}\nL2: Initial={l2_init:.2f}, Personalized={l2_pers:.2f}', 
                    fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
    
    plt.suptitle('Detailed Trajectory Comparison: Original vs Generic vs Personalized ProMPs', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed trajectory plots saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Plot detailed trajectories for tasks p, s, u.")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs='+',
        default=['p', 's', 'u'],
        help="Tasks to plot (default: p s u).",
    )
    parser.add_argument(
        "--generic_library",
        type=str,
        default="generic_library",
        help="Path to generic library directory.",
    )
    parser.add_argument(
        "--user1_personalized",
        type=str,
        default="User_1_ProMP_library_personalized/promp_library_personalized.npz",
        help="Path to User 1's personalized ProMP library.",
    )
    parser.add_argument(
        "--user1_trajectories",
        type=str,
        default="output",
        help="Directory containing User 1's original trajectory CSVs.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="ProMP/vis/tasks_psu_detailed.png",
        help="Output visualization path.",
    )
    parser.add_argument("--n_basis", type=int, default=20, help="RBF basis count.")
    parser.add_argument("--width", type=float, default=0.05, help="RBF width.")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step.")
    args = parser.parse_args()
    
    generic_library_dir = Path(args.generic_library)
    user1_personalized_lib_file = Path(args.user1_personalized)
    user1_trajectories_dir = Path(args.user1_trajectories)
    output_file = Path(args.out)
    
    if not generic_library_dir.exists():
        raise FileNotFoundError(f"Generic library not found: {generic_library_dir}")
    if not user1_personalized_lib_file.exists():
        raise FileNotFoundError(f"User 1 personalized library not found: {user1_personalized_lib_file}")
    
    plot_detailed_tasks(
        args.tasks,
        generic_library_dir,
        user1_personalized_lib_file,
        user1_trajectories_dir,
        output_file,
        args.n_basis,
        args.width,
        args.dt
    )


if __name__ == "__main__":
    main()
