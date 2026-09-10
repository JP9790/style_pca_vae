#!/usr/bin/env python3
"""
Plot detailed trajectory for task T (letter T), using the first demo as original.
Shows: Original (first demo), Generic ProMP, and Personalized ProMP.
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


def main():
    parser = argparse.ArgumentParser(description="Plot detailed trajectory for task T.")
    parser.add_argument(
        "--alphabet_library",
        type=str,
        default="user1_alphabet_generic_library",
        help="Path to alphabet generic library directory.",
    )
    parser.add_argument(
        "--alphabet_trajectories",
        type=str,
        default="user_1_alphabet_library_output",
        help="Directory containing alphabet trajectory CSVs.",
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
        "--out",
        type=str,
        default="ProMP/vis/alphabet_taskt_detailed.png",
        help="Output path for the detailed plot.",
    )
    parser.add_argument("--n_basis", type=int, default=20, help="RBF basis count.")
    parser.add_argument("--width", type=float, default=0.05, help="RBF width.")
    parser.add_argument("--ridge", type=float, default=1e-6, help="Ridge regularizer.")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step.")
    parser.add_argument(
        "--aggressiveness",
        type=float,
        default=16.0,
        help="Aggressiveness factor for personalization (default: 16.0 for very aggressive).",
    )
    parser.add_argument(
        "--style_cov_scale",
        type=float,
        default=0.7,
        help="Scale factor for style covariance addition (Method 4, default: 0.7 for very aggressive).",
    )
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    script_dir = Path(__file__).resolve().parent.parent
    alphabet_library_dir = (script_dir / args.alphabet_library).resolve()
    alphabet_trajectories_dir = (script_dir / args.alphabet_trajectories).resolve()
    pca_file = (script_dir / args.pca_results).resolve()
    anova_file = (script_dir / args.anova_results).resolve()
    user1_trajectories_dir = (script_dir / args.user1_trajectories).resolve()
    output_file = (script_dir / args.out).resolve()
    
    # Verify files exist
    if not pca_file.exists():
        raise FileNotFoundError(f"PCA results file not found: {pca_file}\n"
                              f"Resolved from: {args.pca_results}\n"
                              f"Script directory: {script_dir}\n"
                              f"Current working directory: {Path.cwd()}")
    if not anova_file.exists():
        raise FileNotFoundError(f"ANOVA results file not found: {anova_file}\n"
                              f"Resolved from: {args.anova_results}\n"
                              f"Script directory: {script_dir}\n"
                              f"Current working directory: {Path.cwd()}")
    
    print("Loading data...")
    
    # Load PCA and ANOVA results
    pca_data = np.load(pca_file)
    pca_mean = pca_data['mean']
    pca_eigvecs = pca_data['eigvecs']
    
    style_indices = get_style_components(anova_file)
    print(f"Found {len(style_indices)} style components")
    
    # Load User 1's trajectories to compute style offset
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
    
    target_weights_arr = np.vstack(target_weights)
    
    # Compute style offset (Method 4: Style in Covariance)
    style_coords, style_mean, V_s = compute_style_coordinates(
        target_weights_arr, pca_mean, pca_eigvecs, style_indices
    )
    style_offset = args.aggressiveness * (V_s @ style_mean)
    style_cov_addition = args.aggressiveness * args.style_cov_scale * (V_s @ V_s.T)
    print(f"Style offset norm: {np.linalg.norm(style_offset):.4f}")
    print(f"Style covariance scale: {args.style_cov_scale}")
    print(f"Method 4: Style in Covariance (Aggressiveness: {args.aggressiveness}x)")
    
    # Load generic ProMP for task T
    task_id = "t"
    promp_file = alphabet_library_dir / f"task_{task_id}_initialpromp" / "promp.npz"
    if not promp_file.exists():
        raise FileNotFoundError(f"ProMP file not found: {promp_file}")
    
    promp_data = np.load(promp_file)
    generic_mean = promp_data['mean']
    personalized_mean = generic_mean + style_offset
    
    # Load first demo for task T
    task_folder = alphabet_trajectories_dir / f"task_{task_id}"
    if not task_folder.exists():
        raise FileNotFoundError(f"Task folder not found: {task_folder}")
    
    # Get all trajectory CSVs for task T and sort them
    demo_files = sorted(list(task_folder.rglob("*-trajectory.csv")))
    if len(demo_files) == 0:
        raise FileNotFoundError(f"No trajectory files found in {task_folder}")
    
    first_demo_file = demo_files[0]
    print(f"Using first demo: {first_demo_file.name}")
    
    time, y = read_trajectory_csv(first_demo_file)
    
    # Reconstruct trajectories
    duration = len(y) * args.dt
    generic_traj = reconstruct_trajectory(generic_mean, args.n_basis, args.width, duration, args.dt)
    personalized_traj = reconstruct_trajectory(personalized_mean, args.n_basis, args.width, duration, args.dt)
    
    # Interpolate to match demo length
    if len(generic_traj) != len(y):
        generic_traj = interpolate_trajectory(generic_traj, len(y))
    if len(personalized_traj) != len(y):
        personalized_traj = interpolate_trajectory(personalized_traj, len(y))
    
    # Apply mirror (across y-axis) and 180-degree rotation transformation
    def mirror_and_rotate_180(traj):
        # Mirror across y-axis: negate x coordinate
        # Then rotate 180 degrees: negate both x and y
        mirrored_rotated = traj.copy()
        mirrored_rotated[:, 0] = -mirrored_rotated[:, 0]  # Mirror across y-axis
        mirrored_rotated[:, 0] = -mirrored_rotated[:, 0]  # Rotate 180 (negate x again)
        mirrored_rotated[:, 1] = -mirrored_rotated[:, 1]  # Rotate 180 (negate y)
        return mirrored_rotated
    
    y_rotated = mirror_and_rotate_180(y)
    generic_traj_rotated = mirror_and_rotate_180(generic_traj)
    personalized_traj_rotated = mirror_and_rotate_180(personalized_traj)
    
    # Calculate metrics (using rotated trajectories)
    l2_gen = calculate_l2_norm(y_rotated, generic_traj_rotated)
    l2_pers = calculate_l2_norm(y_rotated, personalized_traj_rotated)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    ax.plot(y_rotated[:, 0], y_rotated[:, 1], 
           'k-', linewidth=4, label='Original', alpha=0.9, zorder=3)
    ax.plot(generic_traj_rotated[:, 0], generic_traj_rotated[:, 1], 
           'b:', linewidth=3, label='Personalized ProMP', alpha=0.8, zorder=2)
    ax.plot(personalized_traj_rotated[:, 0], personalized_traj_rotated[:, 1], 
           'r--', linewidth=3, label='Generic ProMP', alpha=0.8, zorder=2)
    
    # Mark start point (using rotated coordinates)
    ax.plot(y_rotated[0, 0], y_rotated[0, 1], 
           'go', markersize=15, zorder=5, 
           markeredgecolor='black', markeredgewidth=2)
    
    # Mark end points (using rotated coordinates)
    ax.plot(y_rotated[-1, 0], y_rotated[-1, 1], 
           'ks', markersize=15, zorder=5, 
           markeredgecolor='white', markeredgewidth=2)
    ax.plot(generic_traj_rotated[-1, 0], generic_traj_rotated[-1, 1], 
           'bs', markersize=12, zorder=4, 
           markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(personalized_traj_rotated[-1, 0], personalized_traj_rotated[-1, 1], 
           'rs', markersize=12, zorder=4, 
           markeredgecolor='white', markeredgewidth=1.5)
    
    ax.set_xlabel('X Position', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Position', fontsize=14, fontweight='bold')
    ax.set_title(f'Sample Trajectory Comparison: Letter t',
                fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nDetailed trajectory plot saved to: {output_file}")
    print(f"L2-norm - Generic: {l2_gen:.2f}")
    print(f"L2-norm - Personalized: {l2_pers:.2f}")


if __name__ == "__main__":
    main()
