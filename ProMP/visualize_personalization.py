#!/usr/bin/env python3
"""
Visualize ProMP library personalization results.
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
    from scipy.interpolate import interp1d
    
    if len(traj) == target_length:
        return traj
    
    original_indices = np.linspace(0, len(traj) - 1, len(traj))
    target_indices = np.linspace(0, len(traj) - 1, target_length)
    
    interp_func = interp1d(original_indices, traj, axis=0, kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
    interpolated = interp_func(target_indices)
    
    return interpolated


def calculate_l2_norm(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """
    Calculate L2-norm (Euclidean distance) between two trajectories.
    Trajectories are interpolated to the same length if needed.
    
    Args:
        traj1: Trajectory array of shape (T1, 2)
        traj2: Trajectory array of shape (T2, 2)
        
    Returns:
        L2-norm (scalar)
    """
    # Interpolate to same length (use the longer one)
    max_length = max(len(traj1), len(traj2))
    traj1_interp = interpolate_trajectory(traj1, max_length)
    traj2_interp = interpolate_trajectory(traj2, max_length)
    
    # Calculate L2-norm: sqrt(sum((traj1 - traj2)^2))
    diff = traj1_interp - traj2_interp
    l2_norm = np.sqrt(np.sum(diff ** 2))
    
    return float(l2_norm)


def visualize_personalization(
    original_library_file: Path,
    personalized_library_file: Path,
    summary_file: Path,
    output_file: Path,
    trajectories_dir: Path,
    n_basis: int = 20,
    width: float = 0.05
):
    """Create comprehensive visualization of personalization results."""
    # Load libraries
    orig_lib = np.load(original_library_file)
    pers_lib = np.load(personalized_library_file)
    
    # Load summary
    import json
    with summary_file.open('r') as f:
        summary = json.load(f)
    
    orig_tasks = orig_lib['tasks'].astype(str)
    target_user = summary['target_user_id']
    style_components = summary['style_components']
    style_offset_norm = summary['style_offset_norm']
    
    # Load original trajectories from output folder
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
                    original_trajectories[task_id] = y  # Store x, y coordinates
            except Exception as e:
                print(f"Warning: Skipping {csv_path}: {e}")
                continue
    
    print(f"Loaded {len(original_trajectories)} original trajectories")
    
    # Calculate L2-norms
    print("Calculating L2-norms...")
    l2_norms_original = []  # Original input vs initial learned ProMP
    l2_norms_personalized = []  # Original input vs personalized learned ProMP
    task_labels = []
    
    for task in sorted(orig_tasks):
        if task not in original_trajectories:
            print(f"Warning: No original trajectory found for task {task}, skipping")
            continue
        
        task_labels.append(task)
        original_traj = original_trajectories[task]
        
        # Reconstruct trajectories from ProMP means
        orig_mean = orig_lib[f'task_{task}_mean']
        pers_mean = pers_lib[f'task_{task}_mean']
        
        # Get duration from original trajectory (assume 0.01s time step)
        duration = len(original_traj) * 0.01
        dt = 0.01
        
        # Reconstruct ProMP trajectories
        orig_promp_traj = reconstruct_trajectory(orig_mean, n_basis, width, duration, dt)
        pers_promp_traj = reconstruct_trajectory(pers_mean, n_basis, width, duration, dt)
        
        # Ensure ProMP trajectories match original trajectory length
        if len(orig_promp_traj) != len(original_traj):
            orig_promp_traj = interpolate_trajectory(orig_promp_traj, len(original_traj))
        if len(pers_promp_traj) != len(original_traj):
            pers_promp_traj = interpolate_trajectory(pers_promp_traj, len(original_traj))
        
        # Calculate L2-norms
        l2_orig = calculate_l2_norm(original_traj, orig_promp_traj)
        l2_pers = calculate_l2_norm(original_traj, pers_promp_traj)
        
        l2_norms_original.append(l2_orig)
        l2_norms_personalized.append(l2_pers)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Style offset visualization
    ax1 = plt.subplot(3, 3, 1)
    style_offset = pers_lib['style_offset']
    ax1.bar(range(len(style_offset)), style_offset, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Weight Dimension', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Style Offset Value', fontsize=10, fontweight='bold')
    ax1.set_title(f'Style Offset (norm={style_offset_norm:.4f})', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Plot 2: Style component contributions
    ax2 = plt.subplot(3, 3, 2)
    style_comp_indices = pers_lib['style_component_indices']
    pca_data = np.load('ProMP/promp_pca_results.npz')
    pca_eigvecs = pca_data['eigvecs']
    pca_mean = pca_data['mean']
    
    # Get style subspace
    V_s = pca_eigvecs[:, style_comp_indices]  # (P, d_s)
    
    # Project style offset onto style components
    style_coords = V_s.T @ style_offset  # (d_s,)
    
    ax2.bar(range(len(style_coords)), style_coords, alpha=0.7, 
           color='teal', edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Style Component', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Style Coordinate Value', fontsize=10, fontweight='bold')
    ax2.set_title(f'Style Coordinates (Components {style_components})', fontsize=11, fontweight='bold')
    ax2.set_xticks(range(len(style_components)))
    ax2.set_xticklabels([f'PC{c}' for c in style_components])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Plot 3: Mean weight comparison for a sample task
    sample_task = 'b' if 'b' in orig_tasks else orig_tasks[0]
    ax3 = plt.subplot(3, 3, 3)
    orig_mean = orig_lib[f'task_{sample_task}_mean']
    pers_mean = pers_lib[f'task_{sample_task}_mean']
    diff = pers_mean - orig_mean
    
    x = np.arange(len(orig_mean))
    width_bar = 0.35
    ax3.bar(x - width_bar/2, orig_mean, width_bar, label='Original', alpha=0.7, color='blue')
    ax3.bar(x + width_bar/2, pers_mean, width_bar, label='Personalized', alpha=0.7, color='red')
    ax3.set_xlabel('Weight Dimension', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Weight Value', fontsize=10, fontweight='bold')
    ax3.set_title(f'Mean Weights: Task {sample_task}', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Trajectory comparison for sample tasks
    ax4 = plt.subplot(3, 3, 4)
    sample_tasks = orig_tasks[:min(6, len(orig_tasks))]  # Show first 6 tasks
    
    for task in sample_tasks:
        orig_mean = orig_lib[f'task_{task}_mean']
        pers_mean = pers_lib[f'task_{task}_mean']
        
        # Reconstruct trajectories
        orig_traj = reconstruct_trajectory(orig_mean, n_basis, width)
        pers_traj = reconstruct_trajectory(pers_mean, n_basis, width)
        
        # Plot original
        ax4.plot(orig_traj[:, 0], orig_traj[:, 1], 'b-', alpha=0.3, linewidth=1.5, label='Original' if task == sample_tasks[0] else '')
        # Plot personalized
        ax4.plot(pers_traj[:, 0], pers_traj[:, 1], 'r--', alpha=0.5, linewidth=1.5, label='Personalized' if task == sample_tasks[0] else '')
    
    ax4.set_xlabel('X Position', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Y Position', fontsize=10, fontweight='bold')
    ax4.set_title('Trajectory Comparison (Sample Tasks)', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    # Plot 5: Mean weight norm comparison across all tasks
    ax5 = plt.subplot(3, 3, 5)
    orig_norms = []
    pers_norms = []
    diff_norms = []
    
    for task in sorted(orig_tasks):
        orig_mean = orig_lib[f'task_{task}_mean']
        pers_mean = pers_lib[f'task_{task}_mean']
        orig_norms.append(np.linalg.norm(orig_mean))
        pers_norms.append(np.linalg.norm(pers_mean))
        diff_norms.append(np.linalg.norm(pers_mean - orig_mean))
    
    x_pos = np.arange(len(orig_tasks))
    width_bar = 0.25
    ax5.bar(x_pos - width_bar, orig_norms, width_bar, label='Original', alpha=0.7, color='blue')
    ax5.bar(x_pos, pers_norms, width_bar, label='Personalized', alpha=0.7, color='red')
    ax5.bar(x_pos + width_bar, diff_norms, width_bar, label='Difference', alpha=0.7, color='green')
    ax5.set_xlabel('Task', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Mean Weight Norm', fontsize=10, fontweight='bold')
    ax5.set_title('Mean Weight Norms Across Tasks', fontsize=11, fontweight='bold')
    ax5.set_xticks(x_pos[::max(1, len(orig_tasks)//10)])
    ax5.set_xticklabels([orig_tasks[i] for i in range(0, len(orig_tasks), max(1, len(orig_tasks)//10))], rotation=45)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: L2-norm comparison (Original input vs ProMP)
    ax6 = plt.subplot(3, 3, 6)
    if len(task_labels) > 0:
        x_pos = np.arange(len(task_labels))
        width_bar = 0.35
        bars1 = ax6.bar(x_pos - width_bar/2, l2_norms_original, width_bar, 
                       label='Original Input vs Initial ProMP', alpha=0.7, color='blue', edgecolor='black')
        bars2 = ax6.bar(x_pos + width_bar/2, l2_norms_personalized, width_bar, 
                       label='Original Input vs Personalized ProMP', alpha=0.7, color='red', edgecolor='black')
        ax6.set_xlabel('Task', fontsize=10, fontweight='bold')
        ax6.set_ylabel('L2-norm', fontsize=10, fontweight='bold')
        ax6.set_title('L2-norm: Original Trajectories vs ProMP', fontsize=11, fontweight='bold')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(task_labels, rotation=45, ha='right')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=7)
    else:
        ax6.text(0.5, 0.5, 'No original trajectories found', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('L2-norm Comparison', fontsize=11, fontweight='bold')
    
    # Plot 7: Detailed trajectory comparison for task b (with original input)
    ax7 = plt.subplot(3, 3, 7)
    task = 'b' if 'b' in orig_tasks else orig_tasks[0]
    orig_mean = orig_lib[f'task_{task}_mean']
    pers_mean = pers_lib[f'task_{task}_mean']
    
    # Reconstruct ProMP trajectories
    duration = 1.0  # Default duration
    dt = 0.01
    orig_promp_traj = reconstruct_trajectory(orig_mean, n_basis, width, duration, dt)
    pers_promp_traj = reconstruct_trajectory(pers_mean, n_basis, width, duration, dt)
    
    # Plot original input trajectory if available
    if task in original_trajectories:
        original_input_traj = original_trajectories[task]
        # Interpolate to match ProMP trajectory length for better visualization
        if len(original_input_traj) != len(orig_promp_traj):
            original_input_traj = interpolate_trajectory(original_input_traj, len(orig_promp_traj))
        ax7.plot(original_input_traj[:, 0], original_input_traj[:, 1], 
                'k-', linewidth=3, label='Original Input', alpha=0.9, zorder=3)
        ax7.plot(original_input_traj[0, 0], original_input_traj[0, 1], 
                'go', markersize=12, label='Start', zorder=5, markeredgecolor='black', markeredgewidth=1.5)
        ax7.plot(original_input_traj[-1, 0], original_input_traj[-1, 1], 
                'ks', markersize=12, label='End (Input)', zorder=5, markeredgecolor='white', markeredgewidth=1.5)
    
    # Plot ProMP trajectories
    ax7.plot(orig_promp_traj[:, 0], orig_promp_traj[:, 1], 
            'b-', linewidth=2.5, label='Initial ProMP', alpha=0.8, zorder=2)
    ax7.plot(pers_promp_traj[:, 0], pers_promp_traj[:, 1], 
            'r--', linewidth=2.5, label='Personalized ProMP', alpha=0.8, zorder=2)
    
    if task not in original_trajectories:
        # If no original input, show ProMP endpoints
        ax7.plot(orig_promp_traj[0, 0], orig_promp_traj[0, 1], 'go', markersize=10, label='Start', zorder=4)
        ax7.plot(orig_promp_traj[-1, 0], orig_promp_traj[-1, 1], 'bs', markersize=10, label='End (Initial)', zorder=4)
        ax7.plot(pers_promp_traj[-1, 0], pers_promp_traj[-1, 1], 'rs', markersize=10, label='End (Personalized)', zorder=4)
    
    ax7.set_xlabel('X Position', fontsize=10, fontweight='bold')
    ax7.set_ylabel('Y Position', fontsize=10, fontweight='bold')
    ax7.set_title(f'Detailed Trajectory Comparison: Task {task}', fontsize=11, fontweight='bold')
    ax7.legend(fontsize=9, loc='best')
    ax7.grid(True, alpha=0.3)
    ax7.set_aspect('equal')
    
    # Plot 8: Weight difference heatmap
    ax8 = plt.subplot(3, 3, 8)
    task = 'b' if 'b' in orig_tasks else orig_tasks[0]
    orig_mean = orig_lib[f'task_{task}_mean']
    pers_mean = pers_lib[f'task_{task}_mean']
    diff = pers_mean - orig_mean
    
    # Reshape to 2D (x, y dimensions)
    d = 2
    diff_2d = diff.reshape(d, n_basis)
    
    im = ax8.imshow(diff_2d, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax8.set_xlabel('Basis Function Index', fontsize=10, fontweight='bold')
    ax8.set_ylabel('Dimension (X, Y)', fontsize=10, fontweight='bold')
    ax8.set_title(f'Weight Difference: Task {task}', fontsize=11, fontweight='bold')
    ax8.set_yticks([0, 1])
    ax8.set_yticklabels(['X', 'Y'])
    plt.colorbar(im, ax=ax8, label='Weight Difference')
    
    # Plot 9: Summary statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
Personalization Summary

Target User: {target_user}
Source Library: User 1
Number of Tasks: {len(orig_tasks)}
Style Components: {style_components}
Style Offset Norm: {style_offset_norm:.4f}

Statistics:
  Avg Original Norm: {np.mean(orig_norms):.2f}
  Avg Personalized Norm: {np.mean(pers_norms):.2f}
  Avg Difference Norm: {np.mean(diff_norms):.4f}
  
All tasks personalized with
consistent style offset.
    """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            family='monospace')
    
    plt.suptitle(f'ProMP Library Personalization for User {target_user}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize ProMP library personalization.")
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
        "--summary",
        type=str,
        default=None,
        help="Path to personalization summary (default: User_{target_user}_ProMP_library_personalized/personalization_summary.json).",
    )
    parser.add_argument(
        "--target_user",
        type=str,
        default="1",
        help="Target user ID.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output visualization path.",
    )
    parser.add_argument(
        "--trajectories_dir",
        type=str,
        default="output",
        help="Directory containing original trajectory CSVs.",
    )
    args = parser.parse_args()
    
    original_library_file = Path(args.original_library)
    
    if args.personalized_library is None:
        personalized_library_file = Path(f"User_{args.target_user}_ProMP_library_personalized/promp_library_personalized.npz")
    else:
        personalized_library_file = Path(args.personalized_library)
    
    if args.summary is None:
        summary_file = Path(f"User_{args.target_user}_ProMP_library_personalized/personalization_summary.json")
    else:
        summary_file = Path(args.summary)
    
    if args.out is None:
        output_file = Path(f"ProMP/vis/personalization_user_{args.target_user}.png")
    else:
        output_file = Path(args.out)
    
    if not original_library_file.exists():
        raise FileNotFoundError(f"Original library not found: {original_library_file}")
    if not personalized_library_file.exists():
        raise FileNotFoundError(f"Personalized library not found: {personalized_library_file}")
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    trajectories_dir = Path(args.trajectories_dir)
    
    visualize_personalization(
        original_library_file,
        personalized_library_file,
        summary_file,
        output_file,
        trajectories_dir
    )


if __name__ == "__main__":
    main()
