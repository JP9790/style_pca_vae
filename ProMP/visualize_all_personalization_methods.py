#!/usr/bin/env python3
"""
Visualize detailed trajectories for all 8 personalization methods.
Each method gets its own subplot showing original, generic, and personalized trajectories.
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


def visualize_all_methods(
    task_id: str,
    methods_dir: Path,
    generic_library_dir: Path,
    original_library_file: Path,
    user1_trajectories_dir: Path,
    output_file: Path,
    n_basis: int = 20,
    width: float = 0.05,
    dt: float = 0.01
):
    """
    Visualize all 8 personalization methods for a specific task.
    """
    # Method names and descriptions
    method_info = {
        'method1_offset': ('Method 1: Simple Style Offset', 'Baseline - constant offset'),
        'method2_scaling_0.5': ('Method 2: Style Scaling (0.5x)', 'Half strength personalization'),
        'method2_scaling_1.5': ('Method 2: Style Scaling (1.5x)', 'Enhanced personalization'),
        'method3_task_specific': ('Method 3: Task-Specific Style', 'Blend global + task style'),
        'method4_cov': ('Method 4: Style in Covariance', 'Modify mean + covariance'),
        'method6_weighted': ('Method 6: Style-Weighted', 'Confidence-based weighting'),
        'method8_multi': ('Method 8: Multi-Component', 'Weighted style components'),
    }
    
    # Load User 1's original trajectory for this task
    print(f"Loading original trajectory for task {task_id}...")
    user1_traj = None
    for csv_path in user1_trajectories_dir.rglob("*-trajectory.csv"):
        user_id = infer_user_id_from_path(csv_path)
        if user_id == "1":
            try:
                time, y = read_trajectory_csv(csv_path)
                if time.size < 2:
                    continue
                task_id_from_path = infer_task_id_from_path(csv_path)
                if task_id_from_path == task_id:
                    user1_traj = y
                    break
            except Exception as e:
                continue
    
    if user1_traj is None:
        raise ValueError(f"No original trajectory found for User 1, Task {task_id}")
    
    # Load generic ProMP
    print(f"Loading generic ProMP for task {task_id}...")
    generic_promp_file = generic_library_dir / f"task_{task_id}_initialpromp" / "promp.npz"
    if not generic_promp_file.exists():
        raise FileNotFoundError(f"Generic ProMP not found: {generic_promp_file}")
    generic_promp = np.load(generic_promp_file)
    generic_mean = generic_promp['mean']
    
    # Load original library ProMP (for reference)
    original_lib = np.load(original_library_file)
    if f'task_{task_id}_mean' in original_lib:
        original_lib_mean = original_lib[f'task_{task_id}_mean']
    else:
        original_lib_mean = generic_mean  # Fallback
    
    # Load all method libraries
    method_files = {}
    for method_name in method_info.keys():
        method_file = methods_dir / f"{method_name}_personalized.npz"
        if method_file.exists():
            method_files[method_name] = np.load(method_file)
        else:
            print(f"Warning: Method file not found: {method_file}")
    
    # Reconstruct trajectories
    duration = len(user1_traj) * dt
    target_length = len(user1_traj)
    
    # Generic trajectory
    generic_traj = reconstruct_trajectory(generic_mean, n_basis, width, duration, dt)
    generic_traj = interpolate_trajectory(generic_traj, target_length)
    
    # Original library trajectory
    original_lib_traj = reconstruct_trajectory(original_lib_mean, n_basis, width, duration, dt)
    original_lib_traj = interpolate_trajectory(original_lib_traj, target_length)
    
    # Method trajectories
    method_trajectories = {}
    for method_name, method_data in method_files.items():
        if f'task_{task_id}_mean' in method_data:
            method_mean = method_data[f'task_{task_id}_mean']
            method_traj = reconstruct_trajectory(method_mean, n_basis, width, duration, dt)
            method_trajectories[method_name] = interpolate_trajectory(method_traj, target_length)
        else:
            print(f"Warning: Task {task_id} not found in {method_name}")
    
    # Create figure with subplots
    n_methods = len(method_info)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Plot each method
    for idx, (method_name, (title, desc)) in enumerate(method_info.items()):
        ax = axes[idx]
        
        if method_name not in method_trajectories:
            ax.text(0.5, 0.5, f'{title}\n{desc}\n\nMethod not available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            continue
        
        method_traj = method_trajectories[method_name]
        
        # Calculate L2-norms
        l2_generic = calculate_l2_norm(user1_traj, generic_traj)
        l2_method = calculate_l2_norm(user1_traj, method_traj)
        
        # Plot trajectories
        ax.plot(user1_traj[:, 0], user1_traj[:, 1], 
               'k-', linewidth=4, label='Original (User 1)', alpha=0.9, zorder=3)
        ax.plot(generic_traj[:, 0], generic_traj[:, 1], 
               'b:', linewidth=2.5, label='Generic ProMP', alpha=0.7, zorder=2)
        ax.plot(method_traj[:, 0], method_traj[:, 1], 
               'r--', linewidth=3, label='Personalized ProMP', alpha=0.8, zorder=2)
        
        # Mark start point
        ax.plot(user1_traj[0, 0], user1_traj[0, 1], 
               'go', markersize=12, zorder=5, markeredgecolor='black', markeredgewidth=2)
        
        # Mark end points
        ax.plot(user1_traj[-1, 0], user1_traj[-1, 1], 
               'ks', markersize=12, zorder=5, markeredgecolor='white', markeredgewidth=2)
        ax.plot(generic_traj[-1, 0], generic_traj[-1, 1], 
               'bs', markersize=10, zorder=4, markeredgecolor='white', markeredgewidth=1.5)
        ax.plot(method_traj[-1, 0], method_traj[-1, 1], 
               'rs', markersize=10, zorder=4, markeredgecolor='white', markeredgewidth=1.5)
        
        ax.set_xlabel('X Position', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=10, fontweight='bold')
        ax.set_title(f'{title}\n{desc}\nL2: Generic={l2_generic:.1f}, Method={l2_method:.1f}', 
                    fontsize=11, fontweight='bold', pad=8)
        ax.legend(fontsize=8, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
    
    plt.suptitle(f'Personalization Methods Comparison: Task {task_id.upper()}\n'
                 f'Original (User 1) vs Generic ProMP vs Personalized ProMPs', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    print(f"Task: {task_id}")
    print(f"Methods visualized: {len(method_trajectories)}")


def main():
    parser = argparse.ArgumentParser(description="Visualize all personalization methods for a task.")
    parser.add_argument(
        "--task",
        type=str,
        default="p",
        help="Task ID to visualize (e.g., 'p', 's', 'b').",
    )
    parser.add_argument(
        "--methods_dir",
        type=str,
        default="ProMP/personalization_methods_comparison",
        help="Directory containing personalized method libraries.",
    )
    parser.add_argument(
        "--generic_library",
        type=str,
        default="generic_library",
        help="Path to generic library directory.",
    )
    parser.add_argument(
        "--original_library",
        type=str,
        default="User_1_ProMP_library/promp_library.npz",
        help="Path to original ProMP library.",
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
        default="ProMP/vis/all_methods_comparison_task_p.png",
        help="Output visualization path.",
    )
    parser.add_argument("--n_basis", type=int, default=20, help="RBF basis count.")
    parser.add_argument("--width", type=float, default=0.05, help="RBF width.")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step.")
    args = parser.parse_args()
    
    visualize_all_methods(
        args.task,
        Path(args.methods_dir),
        Path(args.generic_library),
        Path(args.original_library),
        Path(args.user1_trajectories),
        Path(args.out),
        args.n_basis,
        args.width,
        args.dt
    )


if __name__ == "__main__":
    main()
