#!/usr/bin/env python3
"""
Personalize generic_library ProMPs using User 1's identified style.
Then compare with User 1's original trajectories using L2-norm and IoU.
"""
from __future__ import annotations

import argparse
import json
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
from scipy.spatial.distance import cdist


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


def calculate_trajectory_iou(traj1: np.ndarray, traj2: np.ndarray, threshold: float = 5.0) -> float:
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


def personalize_generic_library_and_compare(
    generic_library_dir: Path,
    user1_personalized_lib_file: Path,
    user1_trajectories_dir: Path,
    output_dir: Path,
    n_basis: int = 20,
    width: float = 0.05,
    dt: float = 0.01,
    iou_threshold: float = 5.0
):
    """
    Personalize generic library ProMPs and compare with User 1 trajectories.
    """
    # Load User 1's style offset from personalized library
    print("Loading User 1's style offset...")
    user1_pers_lib = np.load(user1_personalized_lib_file)
    style_offset = user1_pers_lib['style_offset']
    print(f"Style offset norm: {np.linalg.norm(style_offset):.4f}")
    
    # Load User 1's original trajectories
    print(f"\nLoading User 1's original trajectories from {user1_trajectories_dir}...")
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
    
    print(f"Loaded {len(user1_trajectories)} User 1 trajectories")
    print(f"User 1 has tasks: {sorted(user1_trajectories.keys())}")
    
    # Process each task in generic library
    print("\nProcessing generic library tasks...")
    l2_norms_initial = []
    l2_norms_personalized = []
    iou_initial = []
    iou_personalized = []
    task_labels = []
    
    # Get all task folders in generic_library
    generic_tasks = sorted([d.name.replace("task_", "").replace("_initialpromp", "") 
                          for d in generic_library_dir.iterdir() 
                          if d.is_dir() and d.name.endswith("_initialpromp")])
    
    print(f"Generic library has {len(generic_tasks)} tasks")
    
    for task in generic_tasks:
        # Skip if User 1 doesn't have this task
        if task not in user1_trajectories:
            print(f"  Task {task}: Skipping (User 1 doesn't have this task)")
            continue
        
        task_labels.append(task)
        user1_traj = user1_trajectories[task]
        
        # Load initial ProMP for this task
        promp_file = generic_library_dir / f"task_{task}_initialpromp" / "promp.npz"
        if not promp_file.exists():
            print(f"  Task {task}: Warning - ProMP file not found, skipping")
            task_labels.pop()  # Remove from labels
            continue
        
        initial_promp = np.load(promp_file)
        initial_mean = initial_promp['mean']
        
        # Personalize: add style offset
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
        
        # Calculate L2-norms
        l2_init = calculate_l2_norm(user1_traj, initial_traj)
        l2_pers = calculate_l2_norm(user1_traj, personalized_traj)
        
        # Calculate IoU
        iou_init = calculate_trajectory_iou(user1_traj, initial_traj, iou_threshold)
        iou_pers = calculate_trajectory_iou(user1_traj, personalized_traj, iou_threshold)
        
        l2_norms_initial.append(l2_init)
        l2_norms_personalized.append(l2_pers)
        iou_initial.append(iou_init)
        iou_personalized.append(iou_pers)
        
        print(f"  Task {task}: L2 (init={l2_init:.2f}, pers={l2_pers:.2f}), IoU (init={iou_init:.3f}, pers={iou_pers:.3f})")
    
    # Create plots
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: L2-norm comparison
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    x_pos = np.arange(len(task_labels))
    width_bar = 0.35
    
    bars1 = ax1.bar(x_pos - width_bar/2, l2_norms_initial, width_bar, 
                   label='User 1 Original vs Generic (Initial) ProMP', alpha=0.8, 
                   color='blue', edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x_pos + width_bar/2, l2_norms_personalized, width_bar, 
                   label='User 1 Original vs Personalized ProMP', alpha=0.8, 
                   color='red', edgecolor='black', linewidth=1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax1.set_xlabel('Task', fontsize=14, fontweight='bold')
    ax1.set_ylabel('L2-norm', fontsize=14, fontweight='bold')
    ax1.set_title('L2-norm: User 1 Original Trajectories vs Generic Library ProMPs', 
                 fontsize=16, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(task_labels, rotation=45, ha='right', fontsize=11)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add statistics
    avg_l2_init = np.mean(l2_norms_initial)
    avg_l2_pers = np.mean(l2_norms_personalized)
    improvement_l2 = ((avg_l2_init - avg_l2_pers) / avg_l2_init * 100) if avg_l2_init > 0 else 0
    
    stats_text = f'Average L2-norm:\nInitial: {avg_l2_init:.2f}\nPersonalized: {avg_l2_pers:.2f}\n'
    if improvement_l2 > 0:
        stats_text += f'Improvement: {improvement_l2:.1f}%'
    else:
        stats_text += f'Change: {improvement_l2:.1f}%'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    l2_plot_file = output_dir / "l2_norm_generic_vs_personalized.png"
    plt.savefig(l2_plot_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nL2-norm plot saved to: {l2_plot_file}")
    
    # Plot 2: IoU comparison
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    bars3 = ax2.bar(x_pos - width_bar/2, iou_initial, width_bar, 
                   label='User 1 Original vs Generic (Initial) ProMP', alpha=0.8, 
                   color='blue', edgecolor='black', linewidth=1)
    bars4 = ax2.bar(x_pos + width_bar/2, iou_personalized, width_bar, 
                   label='User 1 Original vs Personalized ProMP', alpha=0.8, 
                   color='red', edgecolor='black', linewidth=1)
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('Task', fontsize=14, fontweight='bold')
    ax2.set_ylabel('IoU (Intersection over Union)', fontsize=14, fontweight='bold')
    ax2.set_title(f'IoU: User 1 Original Trajectories vs Generic Library ProMPs (threshold={iou_threshold})', 
                 fontsize=16, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(task_labels, rotation=45, ha='right', fontsize=11)
    ax2.set_ylim([0, 1.1])
    ax2.legend(fontsize=12, loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add statistics
    avg_iou_init = np.mean(iou_initial)
    avg_iou_pers = np.mean(iou_personalized)
    improvement_iou = ((avg_iou_pers - avg_iou_init) / avg_iou_init * 100) if avg_iou_init > 0 else 0
    
    stats_text = f'Average IoU:\nInitial: {avg_iou_init:.3f}\nPersonalized: {avg_iou_pers:.3f}\n'
    if improvement_iou > 0:
        stats_text += f'Improvement: {improvement_iou:.1f}%'
    else:
        stats_text += f'Change: {improvement_iou:.1f}%'
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    iou_plot_file = output_dir / "iou_generic_vs_personalized.png"
    plt.savefig(iou_plot_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"IoU plot saved to: {iou_plot_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Tasks compared: {len(task_labels)}")
    print(f"Average L2-norm - Initial: {avg_l2_init:.4f}, Personalized: {avg_l2_pers:.4f}")
    print(f"Average IoU - Initial: {avg_iou_init:.4f}, Personalized: {avg_iou_pers:.4f}")
    print(f"L2-norm improvement: {improvement_l2:.2f}%")
    print(f"IoU improvement: {improvement_iou:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Personalize generic library and compare with User 1.")
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
        "--out_dir",
        type=str,
        default="ProMP/vis",
        help="Output directory for plots.",
    )
    parser.add_argument("--n_basis", type=int, default=20, help="RBF basis count.")
    parser.add_argument("--width", type=float, default=0.05, help="RBF width.")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step.")
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=5.0,
        help="Distance threshold for IoU calculation.",
    )
    args = parser.parse_args()
    
    generic_library_dir = Path(args.generic_library)
    user1_personalized_lib_file = Path(args.user1_personalized)
    user1_trajectories_dir = Path(args.user1_trajectories)
    output_dir = Path(args.out_dir)
    
    if not generic_library_dir.exists():
        raise FileNotFoundError(f"Generic library not found: {generic_library_dir}")
    if not user1_personalized_lib_file.exists():
        raise FileNotFoundError(f"User 1 personalized library not found: {user1_personalized_lib_file}")
    
    personalize_generic_library_and_compare(
        generic_library_dir,
        user1_personalized_lib_file,
        user1_trajectories_dir,
        output_dir,
        args.n_basis,
        args.width,
        args.dt,
        args.iou_threshold
    )


if __name__ == "__main__":
    main()
