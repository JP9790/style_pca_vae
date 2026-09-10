#!/usr/bin/env python3
"""
Evaluate all users' digit libraries:
1. Extract each user's style from PCA results
2. Personalize each user's ProMP library
3. Compute L2-norm and IoU for original (first demo) vs generic and personalized
4. Save all results to a single CSV file
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, List, Dict
import pandas as pd

import numpy as np

# Set matplotlib backend
import os
_root_dir = Path(__file__).resolve().parent
_mpl_cache = _root_dir / ".mplcache"
_cache_home = _root_dir / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_home))
os.environ.setdefault("MPLBACKEND", "Agg")

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
    with anova_file.open('r') as f:
        anova_data = json.load(f)
    
    style_components = []
    for result in anova_data['results']:
        if result['label'] in ['style', 'style_candidate']:
            style_components.append(result['component'] - 1)  # 0-indexed
    
    return sorted(style_components)


def compute_style_offset(
    target_weights: np.ndarray,
    pca_mean: np.ndarray,
    pca_eigvecs: np.ndarray,
    style_component_indices: List[int]
) -> np.ndarray:
    """Compute style offset for a target user."""
    if len(style_component_indices) == 0:
        return np.zeros(pca_mean.shape[0])
    
    # V_s: matrix of style principal directions, shape (P, d_s)
    V_s = pca_eigvecs[:, style_component_indices]  # (P, d_s)
    
    # Center target weights: w_i - w_bar
    centered_weights = target_weights - pca_mean  # (N, P)
    
    # Project onto style subspace: z_i^(s) = V_s^T (w_i - w_bar)
    style_coordinates = centered_weights @ V_s  # (N, d_s)
    
    # Average style coordinates: z_bar^(s) = (1/N) sum_i z_i^(s)
    z_bar_s = np.mean(style_coordinates, axis=0)  # (d_s,)
    
    # Reconstruct style offset: s = V_s * z_bar^(s)
    style_offset = V_s @ z_bar_s  # (P,)
    
    return style_offset


def reconstruct_trajectory(
    weights: np.ndarray, n_basis: int, width: float, duration: float, dt: float
) -> np.ndarray:
    """Reconstruct trajectory from ProMP weights."""
    weights_2d = weights.reshape(2, n_basis)
    t = np.arange(0, duration, dt)
    t_norm = normalize_time(t)
    phi = make_rbf_basis(t_norm, n_basis, width)
    y = phi @ weights_2d.T
    return y


def interpolate_trajectory(traj: np.ndarray, target_length: int) -> np.ndarray:
    """Interpolate trajectory to target length."""
    if len(traj) == target_length:
        return traj
    indices = np.linspace(0, len(traj) - 1, target_length)
    interp_func = lambda x: np.array([
        np.interp(x, np.arange(len(traj)), traj[:, i]) for i in range(traj.shape[1])
    ]).T
    return interp_func(indices)


def calculate_l2_norm(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """Calculate L2-norm (Euclidean distance) between trajectories."""
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


def process_user(
    user_id: int,
    trajectories_dir: Path,
    promp_library_dir: Path,
    pca_file: Path,
    anova_file: Path,
    n_basis: int,
    width: float,
    ridge: float,
    dt: float
) -> List[Dict]:
    """Process a single user and return metrics for all tasks."""
    print(f"\n{'='*80}")
    print(f"Processing User {user_id}")
    print(f"{'='*80}")
    
    # Load PCA results
    pca_data = np.load(pca_file)
    pca_mean = pca_data['mean']
    pca_eigvecs = pca_data['eigvecs']
    
    # Get style components
    style_indices = get_style_components(anova_file)
    print(f"Found {len(style_indices)} style components")
    
    # Load user's trajectories and compute weights
    user_trajectories = {}
    target_weights = []
    
    for csv_path in trajectories_dir.rglob("*-trajectory.csv"):
        task_id = infer_task_id_from_path(csv_path)
        if task_id not in user_trajectories:
            user_trajectories[task_id] = []
        user_trajectories[task_id].append(csv_path)
        
        # Also collect for style computation
        try:
            time, y = read_trajectory_csv(csv_path)
            if time.size < 2:
                continue
            w = fit_weights(time, y, n_basis, width, ridge)
            target_weights.append(w)
        except Exception as e:
            continue
    
    if len(target_weights) == 0:
        print(f"Warning: No trajectories found for user {user_id}")
        return []
    
    target_weights_arr = np.vstack(target_weights)
    print(f"Loaded {len(target_weights)} trajectories for style computation")
    
    # Compute style offset
    style_offset = compute_style_offset(
        target_weights_arr, pca_mean, pca_eigvecs, style_indices
    )
    print(f"Style offset norm: {np.linalg.norm(style_offset):.4f}")
    
    # Process each task
    results = []
    for task_id in sorted(user_trajectories.keys()):
        if task_id not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            continue
        
        # Get all demos for this task
        task_demos = sorted(user_trajectories[task_id])
        if len(task_demos) == 0:
            continue
        
        # Load generic ProMP for this task
        generic_promp_file = promp_library_dir / f"task_{task_id}_initialpromp" / "promp.npz"
        if not generic_promp_file.exists():
            print(f"Warning: Generic ProMP not found for task {task_id}")
            continue
        
        generic_promp = np.load(generic_promp_file)
        generic_mean = generic_promp['mean']
        generic_cov = generic_promp['cov']
        
        # Personalize ProMP
        personalized_mean = generic_mean + style_offset
        
        # Compute metrics for all demos
        l2_gen_values = []
        l2_pers_values = []
        iou_gen_values = []
        iou_pers_values = []
        
        for demo_path in task_demos:
            try:
                time, y_original = read_trajectory_csv(demo_path)
                if time.size < 2:
                    continue
            except Exception as e:
                continue
            
            # Reconstruct trajectories
            duration = len(y_original) * dt
            generic_traj = reconstruct_trajectory(generic_mean, n_basis, width, duration, dt)
            personalized_traj = reconstruct_trajectory(personalized_mean, n_basis, width, duration, dt)
            
            # Interpolate to match original length
            if len(generic_traj) != len(y_original):
                generic_traj = interpolate_trajectory(generic_traj, len(y_original))
            if len(personalized_traj) != len(y_original):
                personalized_traj = interpolate_trajectory(personalized_traj, len(y_original))
            
            # Calculate metrics
            l2_gen = calculate_l2_norm(y_original, generic_traj)
            l2_pers = calculate_l2_norm(y_original, personalized_traj)
            iou_gen = calculate_trajectory_iou(y_original, generic_traj)
            iou_pers = calculate_trajectory_iou(y_original, personalized_traj)
            
            l2_gen_values.append(l2_gen)
            l2_pers_values.append(l2_pers)
            iou_gen_values.append(iou_gen)
            iou_pers_values.append(iou_pers)
        
        if len(l2_gen_values) == 0:
            continue
        
        # Compute mean and std across all demos for this task
        l2_gen_mean = np.mean(l2_gen_values)
        l2_gen_std = np.std(l2_gen_values) if len(l2_gen_values) > 1 else 0.0
        l2_pers_mean = np.mean(l2_pers_values)
        l2_pers_std = np.std(l2_pers_values) if len(l2_pers_values) > 1 else 0.0
        iou_gen_mean = np.mean(iou_gen_values)
        iou_gen_std = np.std(iou_gen_values) if len(iou_gen_values) > 1 else 0.0
        iou_pers_mean = np.mean(iou_pers_values)
        iou_pers_std = np.std(iou_pers_values) if len(iou_pers_values) > 1 else 0.0
        
        results.append({
            'User': user_id,
            'Task': task_id,
            'L2_Generic': l2_gen_mean,
            'L2_Generic_Std': l2_gen_std,
            'L2_Personalized': l2_pers_mean,
            'L2_Personalized_Std': l2_pers_std,
            'IoU_Generic': iou_gen_mean,
            'IoU_Generic_Std': iou_gen_std,
            'IoU_Personalized': iou_pers_mean,
            'IoU_Personalized_Std': iou_pers_std,
            'N_Demos': len(l2_gen_values),
        })
        
        print(f"  Task {task_id}: L2 Generic={l2_gen_mean:.2f}±{l2_gen_std:.2f}, Personalized={l2_pers_mean:.2f}±{l2_pers_std:.2f}, "
              f"IoU Generic={iou_gen_mean:.4f}±{iou_gen_std:.4f}, Personalized={iou_pers_mean:.4f}±{iou_pers_std:.4f} (N={len(l2_gen_values)})")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all users' digit libraries with personalization."
    )
    parser.add_argument(
        "--pca_results",
        type=str,
        default="ProMP/promp_pca_results.npz",
        help="Path to PCA results file"
    )
    parser.add_argument(
        "--anova_results",
        type=str,
        default="ProMP/style_identification_results.json",
        help="Path to ANOVA results file"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="ProMP/all_users_digit_evaluation.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--n_basis",
        type=int,
        default=20,
        help="Number of RBF basis functions"
    )
    parser.add_argument(
        "--width",
        type=float,
        default=0.05,
        help="RBF width"
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-6,
        help="Ridge regularizer"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Time step for trajectory reconstruction"
    )
    parser.add_argument(
        "--start_user",
        type=int,
        default=1,
        help="Start user ID (default: 1)"
    )
    parser.add_argument(
        "--end_user",
        type=int,
        default=24,
        help="End user ID (default: 24)"
    )
    args = parser.parse_args()
    
    base_dir = Path(".").resolve()
    pca_file = base_dir / args.pca_results
    anova_file = base_dir / args.anova_results
    output_csv = base_dir / args.output_csv
    
    print("="*80)
    print("EVALUATING ALL USERS' DIGIT LIBRARIES")
    print("="*80)
    print(f"PCA results: {pca_file}")
    print(f"ANOVA results: {anova_file}")
    print(f"Users: {args.start_user} to {args.end_user}")
    print("="*80)
    
    all_results = []
    
    for user_id in range(args.start_user, args.end_user + 1):
        trajectories_dir = base_dir / f"user_{user_id}_digit_library_output"
        promp_library_dir = base_dir / f"user{user_id}_digit_generic_library"
        
        if not trajectories_dir.exists():
            print(f"Warning: Trajectories directory not found for user {user_id}: {trajectories_dir}")
            continue
        
        if not promp_library_dir.exists():
            print(f"Warning: ProMP library directory not found for user {user_id}: {promp_library_dir}")
            continue
        
        user_results = process_user(
            user_id,
            trajectories_dir,
            promp_library_dir,
            pca_file,
            anova_file,
            args.n_basis,
            args.width,
            args.ridge,
            args.dt
        )
        all_results.extend(user_results)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    if len(df) == 0:
        print("\nNo results to save!")
        return
    
    # Create per-task results (one row per user-task combination)
    # Each user will have 10 rows (one for each task 0-9)
    per_task_rows = []
    for user_id in sorted(df['User'].unique()):
        user_df = df[df['User'] == user_id]
        
        # Get all tasks for this user
        for task_id in sorted(user_df['Task'].unique()):
            task_df = user_df[user_df['Task'] == task_id]
            if len(task_df) == 0:
                continue
            
            # Use the first (and only) row result (already aggregated with mean and std)
            per_task_rows.append({
                'User': user_id,
                'Task': task_id,
                'L2_Generic': task_df['L2_Generic'].iloc[0],
                'L2_Generic_Std': task_df['L2_Generic_Std'].iloc[0],
                'L2_Personalized': task_df['L2_Personalized'].iloc[0],
                'L2_Personalized_Std': task_df['L2_Personalized_Std'].iloc[0],
                'IoU_Generic': task_df['IoU_Generic'].iloc[0],
                'IoU_Generic_Std': task_df['IoU_Generic_Std'].iloc[0],
                'IoU_Personalized': task_df['IoU_Personalized'].iloc[0],
                'IoU_Personalized_Std': task_df['IoU_Personalized_Std'].iloc[0],
                'N_Demos': task_df['N_Demos'].iloc[0] if 'N_Demos' in task_df.columns else 1,
            })
    
    # Create DataFrame with per-task results
    per_task_df = pd.DataFrame(per_task_rows)
    
    # Also compute mean and std across tasks for each user
    summary_rows = []
    for user_id in sorted(per_task_df['User'].unique()):
        user_df = per_task_df[per_task_df['User'] == user_id]
        
        summary_rows.append({
            'User': user_id,
            'L2_Generic_Mean': user_df['L2_Generic'].mean(),
            'L2_Generic_Std': user_df['L2_Generic'].std() if len(user_df) > 1 else 0.0,
            'L2_Personalized_Mean': user_df['L2_Personalized'].mean(),
            'L2_Personalized_Std': user_df['L2_Personalized'].std() if len(user_df) > 1 else 0.0,
            'IoU_Generic_Mean': user_df['IoU_Generic'].mean(),
            'IoU_Generic_Std': user_df['IoU_Generic'].std() if len(user_df) > 1 else 0.0,
            'IoU_Personalized_Mean': user_df['IoU_Personalized'].mean(),
            'IoU_Personalized_Std': user_df['IoU_Personalized'].std() if len(user_df) > 1 else 0.0,
            'N_Tasks': len(user_df),
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Use per-task DataFrame as the main output
    output_df = per_task_df
    
    # Save to CSV (per-task results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    
    # Also save summary statistics to a separate file
    summary_csv = output_csv.parent / (output_csv.stem + "_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"\n{'='*80}")
    print(f"Per-task results saved to: {output_csv}")
    print(f"Summary statistics saved to: {summary_csv}")
    print(f"Total rows: {len(output_df)} (one per user-task combination)")
    print(f"Users: {output_df['User'].nunique()}")
    print(f"Tasks per user: {len(output_df) / output_df['User'].nunique():.1f}")
    print("="*80)


if __name__ == "__main__":
    main()
