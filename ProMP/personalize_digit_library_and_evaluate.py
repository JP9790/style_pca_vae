#!/usr/bin/env python3
"""
Personalize user1_digit_generic_library using User 1's identified style,
then evaluate by comparing each demo with generic and personalized ProMPs.
Compute L2-norm and IoU for each demo, then average and std per task.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List, Dict

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

np.seterr(all="ignore")


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
    """Extract task ID from folder structure (e.g., task_0 -> 0)."""
    parts = csv_path.parts
    for part in parts:
        if part.startswith("task_"):
            return part.replace("task_", "")
    return "unknown"


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


def fit_weights(
    time: np.ndarray, y: np.ndarray, n_basis: int, width: float, ridge: float
) -> np.ndarray:
    """Fit ProMP weights from trajectory data."""
    t_norm = normalize_time(time)
    phi = make_rbf_basis(t_norm, n_basis, width)
    k = phi.shape[1]
    d = y.shape[1]
    weights = np.zeros((d, k))
    a = phi.T @ phi + ridge * np.eye(k)
    for dim in range(d):
        b = phi.T @ y[:, dim]
        weights[dim] = np.linalg.solve(a, b)
    return weights.reshape(-1)


def get_style_components(anova_file: Path) -> List[int]:
    """Get style component indices from ANOVA results."""
    import json
    with anova_file.open('r') as f:
        anova_data = json.load(f)
    
    style_components = []
    for result in anova_data['results']:
        if result['label'] in ['style', 'style_candidate']:
            style_components.append(result['component'] - 1)  # 0-indexed
    
    return sorted(style_components)


def compute_style_coordinates(
    target_weights: np.ndarray,
    pca_mean: np.ndarray,
    pca_eigvecs: np.ndarray,
    style_component_indices: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute style coordinates and return style_mean and V_s."""
    if len(style_component_indices) == 0:
        return np.zeros((target_weights.shape[0], 0)), np.zeros(0), np.zeros((target_weights.shape[1], 0))
    
    V_s = pca_eigvecs[:, style_component_indices]
    centered_weights = target_weights - pca_mean
    style_coordinates = centered_weights @ V_s
    style_mean = np.mean(style_coordinates, axis=0)
    
    return style_coordinates, style_mean, V_s


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


def main():
    parser = argparse.ArgumentParser(description="Personalize digit library and evaluate.")
    parser.add_argument(
        "--digit_library",
        type=str,
        default="user1_digit_generic_library",
        help="Path to digit generic library directory.",
    )
    parser.add_argument(
        "--digit_trajectories",
        type=str,
        default="user_1_digit_library_output",
        help="Directory containing digit trajectory CSVs.",
    )
    parser.add_argument(
        "--pca_results",
        type=str,
        default="ProMP/promp_pca_results.npz",
        help="Path to PCA results.",
    )
    parser.add_argument(
        "--anova_results",
        type=str,
        default="ProMP/style_anova_results.json",
        help="Path to ANOVA results.",
    )
    parser.add_argument(
        "--user1_trajectories",
        type=str,
        default="output",
        help="Directory containing User 1's original trajectory CSVs for style computation.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="ProMP/vis",
        help="Output directory for plots.",
    )
    parser.add_argument("--n_basis", type=int, default=20, help="RBF basis count.")
    parser.add_argument("--width", type=float, default=0.05, help="RBF width.")
    parser.add_argument("--ridge", type=float, default=1e-6, help="Ridge regularizer.")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step.")
    parser.add_argument(
        "--aggressiveness",
        type=float,
        default=5.0,
        help="Aggressiveness factor for personalization (default: 5.0 for more aggressive).",
    )
    parser.add_argument(
        "--style_cov_scale",
        type=float,
        default=0.1,
        help="Scale factor for style covariance addition.",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=5.0,
        help="Distance threshold for IoU calculation.",
    )
    args = parser.parse_args()
    
    digit_library_dir = Path(args.digit_library)
    digit_trajectories_dir = Path(args.digit_trajectories)
    pca_file = Path(args.pca_results)
    anova_file = Path(args.anova_results)
    user1_trajectories_dir = Path(args.user1_trajectories)
    output_dir = Path(args.out_dir)
    
    print("="*60)
    print("PERSONALIZING DIGIT LIBRARY WITH USER 1 STYLE")
    print("="*60)
    
    # Step 1: Load PCA and ANOVA results
    print("\nStep 1: Loading PCA and ANOVA results...")
    pca_data = np.load(pca_file)
    pca_mean = pca_data['mean']
    pca_eigvecs = pca_data['eigvecs']
    
    style_indices = get_style_components(anova_file)
    print(f"Found {len(style_indices)} style components: {[i+1 for i in style_indices]}")
    
    # Step 2: Load User 1's trajectories to compute style offset
    print("\nStep 2: Loading User 1's trajectories for style computation...")
    target_weights = []
    for csv_path in user1_trajectories_dir.rglob("*-trajectory.csv"):
        try:
            time, y = read_trajectory_csv(csv_path)
            if time.size < 2:
                continue
            w = fit_weights(time, y, args.n_basis, args.width, args.ridge)
            target_weights.append(w)
        except Exception as e:
            continue
    
    if len(target_weights) == 0:
        raise ValueError("No trajectories found for User 1")
    
    target_weights_arr = np.vstack(target_weights)
    print(f"Loaded {len(target_weights)} trajectories from User 1")
    
    # Compute style offset
    style_coords, style_mean, V_s = compute_style_coordinates(
        target_weights_arr, pca_mean, pca_eigvecs, style_indices
    )
    style_offset = args.aggressiveness * (V_s @ style_mean)
    style_cov_addition = args.aggressiveness * args.style_cov_scale * (V_s @ V_s.T)
    print(f"Style offset norm: {np.linalg.norm(style_offset):.4f}")
    print(f"Style covariance scale: {args.style_cov_scale}")
    print(f"Aggressiveness: {args.aggressiveness}x")
    
    # Step 3: Load generic ProMPs and personalize using Method 4
    print("\nStep 3: Loading and personalizing generic ProMPs (Method 4: Style in Covariance)...")
    generic_means = {}
    personalized_means = {}
    
    for task_id in range(10):
        task_str = str(task_id)
        promp_file = digit_library_dir / f"task_{task_str}_initialpromp" / "promp.npz"
        if promp_file.exists():
            promp_data = np.load(promp_file)
            generic_means[task_str] = promp_data['mean']
            # Method 4: μ_k^styled = μ_k + aggressiveness * V_s * z̄^(s)
            personalized_means[task_str] = generic_means[task_str] + style_offset
            # Note: Covariance modification (Σ_k^styled = Σ_k + style_cov_addition) 
            # is computed but not used for trajectory reconstruction (only mean is used)
            print(f"  Task {task_str}: personalized (Method 4)")
    
    # Step 4: Load all demos for each task and evaluate
    print("\nStep 4: Evaluating with all demos...")
    
    task_stats = {}
    
    for task_id in range(10):
        task_str = str(task_id)
        if task_str not in generic_means:
            continue
        
        # Load all demos for this task
        task_demos = []
        task_folder = digit_trajectories_dir / f"task_{task_str}"
        if task_folder.exists():
            for csv_path in task_folder.rglob("*-trajectory.csv"):
                try:
                    time, y = read_trajectory_csv(csv_path)
                    if time.size < 2:
                        continue
                    task_demos.append((time, y))
                except Exception as e:
                    continue
        
        if len(task_demos) == 0:
            print(f"  Task {task_str}: No demos found")
            continue
        
        print(f"  Task {task_str}: {len(task_demos)} demos")
        
        # Evaluate each demo
        l2_generic_list = []
        l2_personalized_list = []
        iou_generic_list = []
        iou_personalized_list = []
        
        generic_mean = generic_means[task_str]
        personalized_mean = personalized_means[task_str]
        
        for time, y in task_demos:
            # Reconstruct trajectories
            duration = len(y) * args.dt
            generic_traj = reconstruct_trajectory(generic_mean, args.n_basis, args.width, duration, args.dt)
            personalized_traj = reconstruct_trajectory(personalized_mean, args.n_basis, args.width, duration, args.dt)
            
            # Interpolate to match demo length
            if len(generic_traj) != len(y):
                generic_traj = interpolate_trajectory(generic_traj, len(y))
            if len(personalized_traj) != len(y):
                personalized_traj = interpolate_trajectory(personalized_traj, len(y))
            
            # Calculate metrics
            l2_gen = calculate_l2_norm(y, generic_traj)
            l2_pers = calculate_l2_norm(y, personalized_traj)
            iou_gen = calculate_trajectory_iou(y, generic_traj, args.iou_threshold)
            iou_pers = calculate_trajectory_iou(y, personalized_traj, args.iou_threshold)
            
            l2_generic_list.append(l2_gen)
            l2_personalized_list.append(l2_pers)
            iou_generic_list.append(iou_gen)
            iou_personalized_list.append(iou_pers)
        
        # Compute statistics
        task_stats[task_str] = {
            'n_demos': len(task_demos),
            'l2_generic_mean': np.mean(l2_generic_list),
            'l2_generic_std': np.std(l2_generic_list),
            'l2_personalized_mean': np.mean(l2_personalized_list),
            'l2_personalized_std': np.std(l2_personalized_list),
            'iou_generic_mean': np.mean(iou_generic_list),
            'iou_generic_std': np.std(iou_generic_list),
            'iou_personalized_mean': np.mean(iou_personalized_list),
            'iou_personalized_std': np.std(iou_personalized_list),
        }
    
    # Step 5: Create plots
    print("\nStep 5: Creating plots...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = sorted(task_stats.keys(), key=int)
    task_labels = [f"Task {t}" for t in tasks]
    
    # Plot 1: L2-norm comparison with error bars
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    l2_gen_means = [task_stats[t]['l2_generic_mean'] for t in tasks]
    l2_gen_stds = [task_stats[t]['l2_generic_std'] for t in tasks]
    l2_pers_means = [task_stats[t]['l2_personalized_mean'] for t in tasks]
    l2_pers_stds = [task_stats[t]['l2_personalized_std'] for t in tasks]
    
    x_pos = np.arange(len(tasks))
    width_bar = 0.35
    
    bars1 = ax1.bar(x_pos - width_bar/2, l2_gen_means, width_bar, 
                   yerr=l2_gen_stds, label='Generic ProMP', alpha=0.8, 
                   color='blue', edgecolor='black', linewidth=1, capsize=5)
    bars2 = ax1.bar(x_pos + width_bar/2, l2_pers_means, width_bar, 
                   yerr=l2_pers_stds, label='Personalized ProMP', alpha=0.8, 
                   color='red', edgecolor='black', linewidth=1, capsize=5)
    
    ax1.set_xlabel('Task', fontsize=14, fontweight='bold')
    ax1.set_ylabel('L2-norm (Mean ± Std)', fontsize=14, fontweight='bold')
    ax1.set_title(f'L2-norm Comparison: Generic vs Personalized ProMP (Method 4)\n'
                 f'(Aggressiveness: {args.aggressiveness}x, Style Cov Scale: {args.style_cov_scale}, Error bars: ±1 std)',
                 fontsize=16, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(task_labels, fontsize=11)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels
    for i, (gen_mean, pers_mean) in enumerate(zip(l2_gen_means, l2_pers_means)):
        ax1.text(i - width_bar/2, gen_mean + l2_gen_stds[i] + 2, 
                f'{gen_mean:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(i + width_bar/2, pers_mean + l2_pers_stds[i] + 2, 
                f'{pers_mean:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    l2_plot_file = output_dir / "digit_library_l2_norm_comparison.png"
    plt.savefig(l2_plot_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"L2-norm plot saved to: {l2_plot_file}")
    
    # Plot 2: IoU comparison with error bars
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    iou_gen_means = [task_stats[t]['iou_generic_mean'] for t in tasks]
    iou_gen_stds = [task_stats[t]['iou_generic_std'] for t in tasks]
    iou_pers_means = [task_stats[t]['iou_personalized_mean'] for t in tasks]
    iou_pers_stds = [task_stats[t]['iou_personalized_std'] for t in tasks]
    
    bars3 = ax2.bar(x_pos - width_bar/2, iou_gen_means, width_bar, 
                   yerr=iou_gen_stds, label='Generic ProMP', alpha=0.8, 
                   color='blue', edgecolor='black', linewidth=1, capsize=5)
    bars4 = ax2.bar(x_pos + width_bar/2, iou_pers_means, width_bar, 
                   yerr=iou_pers_stds, label='Personalized ProMP', alpha=0.8, 
                   color='red', edgecolor='black', linewidth=1, capsize=5)
    
    ax2.set_xlabel('Task', fontsize=14, fontweight='bold')
    ax2.set_ylabel('IoU (Mean ± Std)', fontsize=14, fontweight='bold')
    ax2.set_title(f'IoU Comparison: Generic vs Personalized ProMP (Method 4)\n'
                 f'(Threshold: {args.iou_threshold}, Aggressiveness: {args.aggressiveness}x, Style Cov Scale: {args.style_cov_scale}, Error bars: ±1 std)',
                 fontsize=16, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(task_labels, fontsize=11)
    ax2.set_ylim([0, 1.1])
    ax2.legend(fontsize=12, loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels
    for i, (gen_mean, pers_mean) in enumerate(zip(iou_gen_means, iou_pers_means)):
        ax2.text(i - width_bar/2, gen_mean + iou_gen_stds[i] + 0.02, 
                f'{gen_mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax2.text(i + width_bar/2, pers_mean + iou_pers_stds[i] + 0.02, 
                f'{pers_mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    iou_plot_file = output_dir / "digit_library_iou_comparison.png"
    plt.savefig(iou_plot_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"IoU plot saved to: {iou_plot_file}")
    
    # Save statistics to CSV files
    import csv
    
    # Save L2-norm statistics
    l2_csv_file = output_dir / "digit_library_l2_norm_statistics.csv"
    with open(l2_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', 'Generic_Mean', 'Generic_Std', 'Personalized_Mean', 'Personalized_Std', 'N_Demos'])
        for task in tasks:
            stats = task_stats[task]
            writer.writerow([
                task,
                f"{stats['l2_generic_mean']:.4f}",
                f"{stats['l2_generic_std']:.4f}",
                f"{stats['l2_personalized_mean']:.4f}",
                f"{stats['l2_personalized_std']:.4f}",
                stats['n_demos']
            ])
    print(f"\nL2-norm statistics saved to: {l2_csv_file}")
    
    # Save IoU statistics
    iou_csv_file = output_dir / "digit_library_iou_statistics.csv"
    with open(iou_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', 'Generic_Mean', 'Generic_Std', 'Personalized_Mean', 'Personalized_Std', 'N_Demos'])
        for task in tasks:
            stats = task_stats[task]
            writer.writerow([
                task,
                f"{stats['iou_generic_mean']:.4f}",
                f"{stats['iou_generic_std']:.4f}",
                f"{stats['iou_personalized_mean']:.4f}",
                f"{stats['iou_personalized_std']:.4f}",
                stats['n_demos']
            ])
    print(f"IoU statistics saved to: {iou_csv_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for task in tasks:
        stats = task_stats[task]
        print(f"\nTask {task} ({stats['n_demos']} demos):")
        print(f"  L2-norm - Generic: {stats['l2_generic_mean']:.2f} ± {stats['l2_generic_std']:.2f}")
        print(f"            Personalized: {stats['l2_personalized_mean']:.2f} ± {stats['l2_personalized_std']:.2f}")
        print(f"  IoU     - Generic: {stats['iou_generic_mean']:.3f} ± {stats['iou_generic_std']:.3f}")
        print(f"            Personalized: {stats['iou_personalized_mean']:.3f} ± {stats['iou_personalized_std']:.3f}")


if __name__ == "__main__":
    main()
