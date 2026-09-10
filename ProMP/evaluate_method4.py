#!/usr/bin/env python3
"""
Evaluate Method 4 (Style in Covariance) personalization:
- Calculate L2-norm and IoU for all tasks
- Create single plots for L2-norm, IoU, and detailed trajectory for task b
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


def evaluate_method4(
    method4_library_file: Path,
    generic_library_dir: Path,
    user1_trajectories_dir: Path,
    output_dir: Path,
    task_b_for_detail: str = "b",
    n_basis: int = 20,
    width: float = 0.05,
    dt: float = 0.01,
    iou_threshold: float = 5.0
):
    """
    Evaluate Method 4 personalization and create plots.
    """
    # Load Method 4 personalized library
    print("Loading Method 4 personalized library...")
    method4_lib = np.load(method4_library_file)
    method4_tasks = method4_lib['tasks'].astype(str)
    
    # Load generic library for comparison
    print("Loading generic library...")
    
    # Load User 1's original trajectories
    print("Loading User 1's original trajectories...")
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
    
    # Process each task
    print("\nProcessing tasks...")
    l2_norms_generic = []
    l2_norms_method4 = []
    iou_generic = []
    iou_method4 = []
    task_labels = []
    
    for task in sorted(method4_tasks):
        # Skip if User 1 doesn't have this task
        if task not in user1_trajectories:
            print(f"  Task {task}: Skipping (User 1 doesn't have this task)")
            continue
        
        task_labels.append(task)
        user1_traj = user1_trajectories[task]
        
        # Load generic ProMP
        generic_promp_file = generic_library_dir / f"task_{task}_initialpromp" / "promp.npz"
        if not generic_promp_file.exists():
            print(f"  Task {task}: Warning - Generic ProMP file not found, skipping")
            task_labels.pop()
            continue
        
        generic_promp = np.load(generic_promp_file)
        generic_mean = generic_promp['mean']
        
        # Load Method 4 ProMP
        if f'task_{task}_mean' not in method4_lib:
            print(f"  Task {task}: Warning - Method 4 ProMP not found, skipping")
            task_labels.pop()
            continue
        
        method4_mean = method4_lib[f'task_{task}_mean']
        
        # Reconstruct trajectories
        duration = len(user1_traj) * dt
        generic_traj = reconstruct_trajectory(generic_mean, n_basis, width, duration, dt)
        method4_traj = reconstruct_trajectory(method4_mean, n_basis, width, duration, dt)
        
        # Interpolate to match User 1 trajectory length
        if len(generic_traj) != len(user1_traj):
            generic_traj = interpolate_trajectory(generic_traj, len(user1_traj))
        if len(method4_traj) != len(user1_traj):
            method4_traj = interpolate_trajectory(method4_traj, len(user1_traj))
        
        # Calculate L2-norms
        l2_gen = calculate_l2_norm(user1_traj, generic_traj)
        l2_m4 = calculate_l2_norm(user1_traj, method4_traj)
        
        # Calculate IoU
        iou_gen = calculate_trajectory_iou(user1_traj, generic_traj, iou_threshold)
        iou_m4 = calculate_trajectory_iou(user1_traj, method4_traj, iou_threshold)
        
        l2_norms_generic.append(l2_gen)
        l2_norms_method4.append(l2_m4)
        iou_generic.append(iou_gen)
        iou_method4.append(iou_m4)
        
        print(f"  Task {task}: L2 (gen={l2_gen:.2f}, m4={l2_m4:.2f}), IoU (gen={iou_gen:.3f}, m4={iou_m4:.3f})")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Plot 1: L2-norm comparison
    # ========================================================================
    print("\nCreating L2-norm comparison plot...")
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    x_pos = np.arange(len(task_labels))
    width_bar = 0.35
    
    bars1 = ax1.bar(x_pos - width_bar/2, l2_norms_generic, width_bar, 
                   label='User 1 Original vs Generic ProMP', alpha=0.8, 
                   color='blue', edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x_pos + width_bar/2, l2_norms_method4, width_bar, 
                   label='User 1 Original vs Method 4 (Style in Covariance)', alpha=0.8, 
                   color='red', edgecolor='black', linewidth=1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax1.set_xlabel('Task', fontsize=14, fontweight='bold')
    ax1.set_ylabel('L2-norm', fontsize=14, fontweight='bold')
    ax1.set_title('L2-norm Comparison: Method 4 (Style in Covariance) Personalization', 
                 fontsize=16, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(task_labels, rotation=45, ha='right', fontsize=11)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add statistics
    avg_l2_gen = np.mean(l2_norms_generic)
    avg_l2_m4 = np.mean(l2_norms_method4)
    improvement_l2 = ((avg_l2_gen - avg_l2_m4) / avg_l2_gen * 100) if avg_l2_gen > 0 else 0
    
    stats_text = f'Average L2-norm:\nGeneric: {avg_l2_gen:.2f}\nMethod 4: {avg_l2_m4:.2f}\n'
    if improvement_l2 > 0:
        stats_text += f'Improvement: {improvement_l2:.1f}%'
    else:
        stats_text += f'Change: {improvement_l2:.1f}%'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    l2_plot_file = output_dir / "method4_l2_norm_comparison.png"
    plt.savefig(l2_plot_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"L2-norm plot saved to: {l2_plot_file}")
    
    # ========================================================================
    # Plot 2: IoU comparison
    # ========================================================================
    print("Creating IoU comparison plot...")
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    bars3 = ax2.bar(x_pos - width_bar/2, iou_generic, width_bar, 
                   label='User 1 Original vs Generic ProMP', alpha=0.8, 
                   color='blue', edgecolor='black', linewidth=1)
    bars4 = ax2.bar(x_pos + width_bar/2, iou_method4, width_bar, 
                   label='User 1 Original vs Method 4 (Style in Covariance)', alpha=0.8, 
                   color='red', edgecolor='black', linewidth=1)
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('Task', fontsize=14, fontweight='bold')
    ax2.set_ylabel('IoU (Intersection over Union)', fontsize=14, fontweight='bold')
    ax2.set_title(f'IoU Comparison: Method 4 (Style in Covariance) Personalization (threshold={iou_threshold})', 
                 fontsize=16, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(task_labels, rotation=45, ha='right', fontsize=11)
    ax2.set_ylim([0, 1.1])
    ax2.legend(fontsize=12, loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add statistics
    avg_iou_gen = np.mean(iou_generic)
    avg_iou_m4 = np.mean(iou_method4)
    improvement_iou = ((avg_iou_m4 - avg_iou_gen) / avg_iou_gen * 100) if avg_iou_gen > 0 else 0
    
    stats_text = f'Average IoU:\nGeneric: {avg_iou_gen:.3f}\nMethod 4: {avg_iou_m4:.3f}\n'
    if improvement_iou > 0:
        stats_text += f'Improvement: {improvement_iou:.1f}%'
    else:
        stats_text += f'Change: {improvement_iou:.1f}%'
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    iou_plot_file = output_dir / "method4_iou_comparison.png"
    plt.savefig(iou_plot_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"IoU plot saved to: {iou_plot_file}")
    
    # ========================================================================
    # Plot 3: Detailed trajectory for task b
    # ========================================================================
    print(f"Creating detailed trajectory plot for task {task_b_for_detail}...")
    
    if task_b_for_detail not in user1_trajectories:
        print(f"Warning: Task {task_b_for_detail} not found in User 1 trajectories")
        return
    
    user1_traj_b = user1_trajectories[task_b_for_detail]
    
    # Load generic ProMP for task b
    generic_promp_file_b = generic_library_dir / f"task_{task_b_for_detail}_initialpromp" / "promp.npz"
    if not generic_promp_file_b.exists():
        print(f"Warning: Generic ProMP not found for task {task_b_for_detail}")
        return
    
    generic_promp_b = np.load(generic_promp_file_b)
    generic_mean_b = generic_promp_b['mean']
    
    # Load Method 4 ProMP for task b
    if f'task_{task_b_for_detail}_mean' not in method4_lib:
        print(f"Warning: Method 4 ProMP not found for task {task_b_for_detail}")
        return
    
    method4_mean_b = method4_lib[f'task_{task_b_for_detail}_mean']
    
    # Reconstruct trajectories
    duration_b = len(user1_traj_b) * dt
    generic_traj_b = reconstruct_trajectory(generic_mean_b, n_basis, width, duration_b, dt)
    method4_traj_b = reconstruct_trajectory(method4_mean_b, n_basis, width, duration_b, dt)
    
    # Interpolate to match User 1 trajectory length
    if len(generic_traj_b) != len(user1_traj_b):
        generic_traj_b = interpolate_trajectory(generic_traj_b, len(user1_traj_b))
    if len(method4_traj_b) != len(user1_traj_b):
        method4_traj_b = interpolate_trajectory(method4_traj_b, len(user1_traj_b))
    
    # Calculate metrics
    l2_gen_b = calculate_l2_norm(user1_traj_b, generic_traj_b)
    l2_m4_b = calculate_l2_norm(user1_traj_b, method4_traj_b)
    iou_gen_b = calculate_trajectory_iou(user1_traj_b, generic_traj_b, iou_threshold)
    iou_m4_b = calculate_trajectory_iou(user1_traj_b, method4_traj_b, iou_threshold)
    
    # Create detailed plot
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    
    ax3.plot(user1_traj_b[:, 0], user1_traj_b[:, 1], 
            'k-', linewidth=4, label='Original (User 1)', alpha=0.9, zorder=3)
    ax3.plot(generic_traj_b[:, 0], generic_traj_b[:, 1], 
            'b:', linewidth=3, label='Generic ProMP', alpha=0.8, zorder=2)
    ax3.plot(method4_traj_b[:, 0], method4_traj_b[:, 1], 
            'r--', linewidth=3, label='Method 4 (Style in Covariance)', alpha=0.8, zorder=2)
    
    # Mark start point
    ax3.plot(user1_traj_b[0, 0], user1_traj_b[0, 1], 
            'go', markersize=15, label='Start', zorder=5, 
            markeredgecolor='black', markeredgewidth=2)
    
    # Mark end points
    ax3.plot(user1_traj_b[-1, 0], user1_traj_b[-1, 1], 
            'ks', markersize=15, label='End (Original)', zorder=5, 
            markeredgecolor='white', markeredgewidth=2)
    ax3.plot(generic_traj_b[-1, 0], generic_traj_b[-1, 1], 
            'bs', markersize=12, label='End (Generic)', zorder=4, 
            markeredgecolor='white', markeredgewidth=1.5)
    ax3.plot(method4_traj_b[-1, 0], method4_traj_b[-1, 1], 
            'rs', markersize=12, label='End (Method 4)', zorder=4, 
            markeredgecolor='white', markeredgewidth=1.5)
    
    ax3.set_xlabel('X Position', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Y Position', fontsize=12, fontweight='bold')
    ax3.set_title(f'Detailed Trajectory Comparison: Task {task_b_for_detail.upper()}\n'
                 f'Method 4 (Style in Covariance) Personalization\n'
                 f'L2-norm: Generic={l2_gen_b:.2f}, Method 4={l2_m4_b:.2f} | '
                 f'IoU: Generic={iou_gen_b:.3f}, Method 4={iou_m4_b:.3f}',
                 fontsize=14, fontweight='bold', pad=15)
    ax3.legend(fontsize=11, loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    detail_plot_file = output_dir / f"method4_detailed_task_{task_b_for_detail}.png"
    plt.savefig(detail_plot_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Detailed trajectory plot saved to: {detail_plot_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("METHOD 4 EVALUATION SUMMARY")
    print("="*60)
    print(f"Tasks compared: {len(task_labels)}")
    print(f"Average L2-norm - Generic: {avg_l2_gen:.4f}, Method 4: {avg_l2_m4:.4f}")
    print(f"Average IoU - Generic: {avg_iou_gen:.4f}, Method 4: {avg_iou_m4:.4f}")
    print(f"L2-norm improvement: {improvement_l2:.2f}%")
    print(f"IoU improvement: {improvement_iou:.2f}%")
    print(f"\nTask {task_b_for_detail.upper()} details:")
    print(f"  L2-norm - Generic: {l2_gen_b:.4f}, Method 4: {l2_m4_b:.4f}")
    print(f"  IoU - Generic: {iou_gen_b:.4f}, Method 4: {iou_m4_b:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Method 4 personalization.")
    parser.add_argument(
        "--method4_library",
        type=str,
        default="ProMP/personalization_methods_comparison/method4_cov_personalized.npz",
        help="Path to Method 4 personalized library.",
    )
    parser.add_argument(
        "--generic_library",
        type=str,
        default="generic_library",
        help="Path to generic library directory.",
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
    parser.add_argument(
        "--task_b",
        type=str,
        default="b",
        help="Task ID for detailed trajectory plot.",
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
    
    evaluate_method4(
        Path(args.method4_library),
        Path(args.generic_library),
        Path(args.user1_trajectories),
        Path(args.out_dir),
        args.task_b,
        args.n_basis,
        args.width,
        args.dt,
        args.iou_threshold
    )


if __name__ == "__main__":
    main()
