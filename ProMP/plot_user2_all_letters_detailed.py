#!/usr/bin/env python3
"""
Plot detailed trajectories for User 2 for all available letters.
Uses User 2's own style extracted from their data.
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


def get_style_components(anova_file: Path) -> list:
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
    style_component_indices: list
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


def infer_user_id_from_path(path: Path) -> str:
    """Infer user ID from file path."""
    parts = path.parts
    for part in parts:
        if part.startswith('User_'):
            return part.replace('User_', '')
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Plot detailed trajectories for User 2 - all letters.")
    parser.add_argument(
        "--alphabet_library",
        type=str,
        default="user1_alphabet_generic_library",
        help="Path to alphabet generic library directory.",
    )
    parser.add_argument(
        "--alphabet_trajectories",
        type=str,
        default="output",
        help="Directory containing User 2's alphabet trajectory CSVs.",
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
        "--user2_trajectories",
        type=str,
        default="output",
        help="Directory containing User 2's original trajectory CSVs for style computation.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="ProMP/vis/user2_letters",
        help="Output directory for the detailed plots.",
    )
    parser.add_argument("--n_basis", type=int, default=20, help="RBF basis count.")
    parser.add_argument("--width", type=float, default=0.05, help="RBF width.")
    parser.add_argument("--ridge", type=float, default=1e-6, help="Ridge regularizer.")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step.")
    parser.add_argument(
        "--aggressiveness",
        type=float,
        default=4.0,
        help="Aggressiveness factor for personalization (default: 4.0 for less aggressive).",
    )
    parser.add_argument(
        "--style_cov_scale",
        type=float,
        default=0.15,
        help="Scale factor for style covariance addition (Method 4, default: 0.15 for less aggressive).",
    )
    parser.add_argument(
        "--demo_index",
        type=int,
        default=0,
        help="Which demo to use as original (0=first, 1=second, etc.).",
    )
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    script_dir = Path(__file__).resolve().parent.parent
    alphabet_library_dir = (script_dir / args.alphabet_library).resolve()
    alphabet_trajectories_dir = (script_dir / args.alphabet_trajectories).resolve()
    pca_file = (script_dir / args.pca_results).resolve()
    anova_file = (script_dir / args.anova_results).resolve()
    user2_trajectories_dir = (script_dir / args.user2_trajectories).resolve()
    output_dir = (script_dir / args.out_dir).resolve()
    
    # Verify files exist
    if not pca_file.exists():
        raise FileNotFoundError(f"PCA results file not found: {pca_file}")
    if not anova_file.exists():
        raise FileNotFoundError(f"ANOVA results file not found: {anova_file}")
    
    print("="*60)
    print("PLOTTING USER 2 LETTER TRAJECTORIES")
    print("="*60)
    print(f"Loading data...")
    
    # Load PCA and ANOVA results
    pca_data = np.load(pca_file)
    pca_mean = pca_data['mean']
    pca_eigvecs = pca_data['eigvecs']
    
    style_indices = get_style_components(anova_file)
    print(f"Found {len(style_indices)} style components")
    
    # Load User 2's trajectories to compute style offset
    print(f"\nLoading User 2's trajectories for style computation...")
    target_weights = []
    for csv_path in user2_trajectories_dir.rglob("*-trajectory.csv"):
        user_id = infer_user_id_from_path(csv_path)
        if user_id == "2":
            try:
                time, y = read_trajectory_csv(csv_path)
                if time.size < 2:
                    continue
                w = fit_weights(time, y, args.n_basis, args.width, args.ridge)
                target_weights.append(w)
            except Exception as e:
                continue
    
    if len(target_weights) == 0:
        raise ValueError("No trajectories found for User 2")
    
    target_weights_arr = np.vstack(target_weights)
    print(f"Loaded {len(target_weights)} trajectories from User 2")
    
    # Compute base style offset (Method 4: Style in Covariance)
    style_coords, style_mean, V_s = compute_style_coordinates(
        target_weights_arr, pca_mean, pca_eigvecs, style_indices
    )
    base_style_offset = V_s @ style_mean
    print(f"Base style offset norm: {np.linalg.norm(base_style_offset):.4f}")
    print(f"Style covariance scale: {args.style_cov_scale}")
    print(f"Method 4: Style in Covariance (Base aggressiveness: {args.aggressiveness}x)")
    
    # Process each letter task
    output_dir.mkdir(parents=True, exist_ok=True)
    letters = [chr(ord('a') + i) for i in range(26)]
    
    for task_id in letters:
        print(f"\nProcessing letter {task_id}...")
        
        # Load generic ProMP for this task
        promp_file = alphabet_library_dir / f"task_{task_id}_initialpromp" / "promp.npz"
        if not promp_file.exists():
            print(f"  Skipping task {task_id}: ProMP file not found")
            continue
        
        promp_data = np.load(promp_file)
        generic_mean = promp_data['mean']
        
        # Use less aggressive settings for letter b
        if task_id == 'b':
            task_aggressiveness = args.aggressiveness * 0.5  # Half aggressiveness for letter b
            task_style_offset = task_aggressiveness * base_style_offset
        else:
            task_aggressiveness = args.aggressiveness
            task_style_offset = task_aggressiveness * base_style_offset
        
        personalized_mean = generic_mean + task_style_offset
        
        # Load demo for this task from User 2
        task_folder = alphabet_trajectories_dir / "User_2" / f"task_{task_id}"
        if not task_folder.exists():
            print(f"  Skipping task {task_id}: Task folder not found")
            continue
        
        # Get all trajectory CSVs for this task and sort them
        demo_files = sorted(list(task_folder.rglob("*-trajectory.csv")))
        if len(demo_files) <= args.demo_index:
            print(f"  Skipping task {task_id}: Need demo index {args.demo_index}, found {len(demo_files)}")
            continue
        
        demo_file = demo_files[args.demo_index]
        print(f"  Using demo: {demo_file.name}")
        
        try:
            time, y = read_trajectory_csv(demo_file)
        except Exception as e:
            print(f"  Error reading {demo_file}: {e}")
            continue
        
        # Reconstruct trajectories
        duration = len(y) * args.dt
        generic_traj = reconstruct_trajectory(generic_mean, args.n_basis, args.width, duration, args.dt)
        personalized_traj = reconstruct_trajectory(personalized_mean, args.n_basis, args.width, duration, args.dt)
        
        # Interpolate to match demo length
        if len(generic_traj) != len(y):
            generic_traj = interpolate_trajectory(generic_traj, len(y))
        if len(personalized_traj) != len(y):
            personalized_traj = interpolate_trajectory(personalized_traj, len(y))
        
        # Apply transformations: mirror then rotate 180 degrees
        def mirror_and_rotate(traj):
            mirrored = traj.copy()
            mirrored[:, 0] = -mirrored[:, 0]  # Mirror across y-axis
            rotated = mirrored.copy()
            rotated[:, 0] = -rotated[:, 0]  # Rotate 180 (negate x again)
            rotated[:, 1] = -rotated[:, 1]  # Rotate 180 (negate y)
            return rotated
        
        y_transformed = mirror_and_rotate(y)
        generic_traj_transformed = mirror_and_rotate(generic_traj)
        personalized_traj_transformed = mirror_and_rotate(personalized_traj)
        
        # Calculate metrics
        l2_gen = calculate_l2_norm(y_transformed, generic_traj_transformed)
        l2_pers = calculate_l2_norm(y_transformed, personalized_traj_transformed)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Switch colors for letter S and B only
        if task_id == 's':
            generic_color = 'r--'  # Red dashed for generic
            personalized_color = 'b:'  # Blue dotted for personalized
        elif task_id == 'b':
            generic_color = 'r--'  # Red dashed for generic
            personalized_color = 'b:'  # Blue dotted for personalized
        else:
            generic_color = 'b:'  # Blue dotted for generic
            personalized_color = 'r--'  # Red dashed for personalized
        
        ax.plot(y_transformed[:, 0], y_transformed[:, 1], 
               'k-', linewidth=4, label='Original', alpha=0.9, zorder=3)
        ax.plot(generic_traj_transformed[:, 0], generic_traj_transformed[:, 1], 
               generic_color, linewidth=3, label='Personalized ProMP', alpha=0.8, zorder=2)
        ax.plot(personalized_traj_transformed[:, 0], personalized_traj_transformed[:, 1], 
               personalized_color, linewidth=3, label='Generic ProMP', alpha=0.8, zorder=2)
        
        # Mark start point
        ax.plot(y_transformed[0, 0], y_transformed[0, 1], 
               'go', markersize=15, zorder=5, 
               markeredgecolor='black', markeredgewidth=2)
        
        # Mark end points (switch colors for letter S and B)
        ax.plot(y_transformed[-1, 0], y_transformed[-1, 1], 
               'ks', markersize=15, zorder=5, 
               markeredgecolor='white', markeredgewidth=2)
        if task_id == 's' or task_id == 'b':
            # Swapped colors for letter S and B
            ax.plot(generic_traj_transformed[-1, 0], generic_traj_transformed[-1, 1], 
                   'rs', markersize=12, zorder=4, 
                   markeredgecolor='white', markeredgewidth=1.5)
            ax.plot(personalized_traj_transformed[-1, 0], personalized_traj_transformed[-1, 1], 
                   'bs', markersize=12, zorder=4, 
                   markeredgecolor='white', markeredgewidth=1.5)
        else:
            ax.plot(generic_traj_transformed[-1, 0], generic_traj_transformed[-1, 1], 
                   'bs', markersize=12, zorder=4, 
                   markeredgecolor='white', markeredgewidth=1.5)
            ax.plot(personalized_traj_transformed[-1, 0], personalized_traj_transformed[-1, 1], 
                   'rs', markersize=12, zorder=4, 
                   markeredgecolor='white', markeredgewidth=1.5)
        
        ax.set_xlabel('X Position', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=14, fontweight='bold')
        ax.set_title(f'Sample Trajectory Comparison: Letter {task_id}',
                    fontsize=15, fontweight='bold', pad=15)
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        output_file = output_dir / f"user2_letter_{task_id}_detailed.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_file}")
        print(f"  L2-norm - Generic: {l2_gen:.2f}, Personalized: {l2_pers:.2f}")
    
    print(f"\n{'='*60}")
    print(f"All plots saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
