#!/usr/bin/env python3
"""
Personalize a ProMP library using identified style components.

Following the paper description:
1. Given style-related principal components V_s, project target user's weights onto style subspace
2. Average style coordinates to get single style representation
3. Reconstruct style offset in original weight space
4. Add style offset to each task-specific ProMP mean
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

np.seterr(all="ignore")


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
    """
    Fit ProMP weights from trajectory data.
    Returns flattened weight vector of shape (d * n_basis,).
    """
    t_norm = normalize_time(time)
    phi = make_rbf_basis(t_norm, n_basis, width)  # (T, K)
    k = phi.shape[1]
    d = y.shape[1]
    weights = np.zeros((d, k))
    a = phi.T @ phi + ridge * np.eye(k)
    for dim in range(d):
        b = phi.T @ y[:, dim]
        weights[dim] = np.linalg.solve(a, b)
    return weights.reshape(-1)


def get_style_components(anova_file: Path, style_labels: List[str] = None) -> List[int]:
    """
    Get style component indices from ANOVA results.
    
    Args:
        anova_file: Path to style_anova_results.json
        style_labels: List of labels to consider as style (default: ['style', 'style_candidate'])
        
    Returns:
        List of component indices (0-indexed)
    """
    if style_labels is None:
        style_labels = ['style', 'style_candidate']
    
    with anova_file.open('r') as f:
        anova_data = json.load(f)
    
    style_components = []
    for result in anova_data['results']:
        if result['label'] in style_labels:
            # Convert from 1-indexed to 0-indexed
            style_components.append(result['component'] - 1)
    
    return sorted(style_components)


def compute_style_offset(
    target_weights: np.ndarray,
    pca_mean: np.ndarray,
    pca_eigvecs: np.ndarray,
    style_component_indices: List[int]
) -> np.ndarray:
    """
    Compute style offset for a target user.
    
    Following the paper:
    1. Project weights onto style subspace: z_i^(s) = V_s^T (w_i - w_bar)
    2. Average style coordinates: z_bar^(s) = (1/N) sum_i z_i^(s)
    3. Reconstruct style offset: s = V_s * z_bar^(s)
    
    Args:
        target_weights: Array of shape (N, P) - weights from target user
        pca_mean: Mean weight vector from PCA, shape (P,)
        pca_eigvecs: Principal directions from PCA, shape (P, P) - each column is a PC
        style_component_indices: List of component indices (0-indexed) that are style-related
        
    Returns:
        Style offset vector of shape (P,)
    """
    if len(style_component_indices) == 0:
        print("Warning: No style components found, returning zero offset")
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


def personalize_library(
    library_file: Path,
    pca_file: Path,
    anova_file: Path,
    target_user_id: str,
    output_dir: Path,
    output_dir_trajectories: Path,
    n_basis: int = 20,
    width: float = 0.05,
    ridge: float = 1e-6
) -> None:
    """
    Personalize a ProMP library for a target user using style components.
    
    Args:
        library_file: Path to source ProMP library (e.g., User_1_ProMP_library)
        pca_file: Path to PCA results
        anova_file: Path to ANOVA results (to identify style components)
        target_user_id: User ID to personalize for
        output_dir: Directory to save personalized library
        output_dir_trajectories: Directory containing target user's trajectories
        n_basis, width, ridge: ProMP parameters (must match library parameters)
    """
    # Load library
    print(f"Loading ProMP library from {library_file}...")
    library_data = np.load(library_file)
    library_tasks = library_data['tasks'].astype(str)
    
    # Extract task means and covariances
    library = {}
    for task in library_tasks:
        library[task] = {
            'mean': library_data[f'task_{task}_mean'],
            'cov': library_data[f'task_{task}_cov'],
        }
    
    print(f"Library contains {len(library)} tasks")
    
    # Load PCA results
    print(f"Loading PCA results from {pca_file}...")
    pca_data = np.load(pca_file)
    pca_mean = pca_data['mean']  # w_bar, shape (P,)
    pca_eigvecs = pca_data['eigvecs']  # Principal directions, shape (P, P)
    
    # Get style component indices
    print(f"Loading style components from {anova_file}...")
    style_component_indices = get_style_components(anova_file)
    print(f"Found {len(style_component_indices)} style components: {[i+1 for i in style_component_indices]}")
    
    if len(style_component_indices) == 0:
        print("Warning: No style components found. Cannot personalize library.")
        return
    
    # Load target user's trajectories and compute weights
    print(f"\nLoading target user {target_user_id} trajectories...")
    target_weights = []
    target_tasks = []
    
    for csv_path in output_dir_trajectories.rglob("*-trajectory.csv"):
        user_id = infer_user_id_from_path(csv_path)
        if user_id == target_user_id:
            try:
                time, y = read_trajectory_csv(csv_path)
                if time.size < 2:
                    continue
                task_id = infer_task_id_from_path(csv_path)
                w = fit_weights(time, y, n_basis, width, ridge)
                target_weights.append(w)
                target_tasks.append(task_id)
            except Exception as e:
                print(f"Warning: Skipping {csv_path}: {e}")
                continue
    
    if len(target_weights) == 0:
        raise ValueError(f"No trajectories found for user {target_user_id}")
    
    target_weights_arr = np.vstack(target_weights)  # (N, P)
    print(f"Loaded {len(target_weights)} trajectories from user {target_user_id}")
    
    # Compute style offset
    print("\nComputing style offset...")
    style_offset = compute_style_offset(
        target_weights_arr,
        pca_mean,
        pca_eigvecs,
        style_component_indices
    )
    print(f"Style offset magnitude: {np.linalg.norm(style_offset):.6f}")
    
    # Personalize library: mu_k^styled = mu_k + s
    print("\nPersonalizing ProMP library...")
    personalized_library = {}
    for task_id, task_data in library.items():
        original_mean = task_data['mean']
        personalized_mean = original_mean + style_offset
        personalized_library[task_id] = {
            'mean': personalized_mean,
            'cov': task_data['cov'],  # Covariance unchanged
            'original_mean': original_mean,
            'style_offset': style_offset,
        }
        print(f"  Task {task_id}: personalized")
    
    # Save personalized library
    output_dir.mkdir(parents=True, exist_ok=True)
    library_file_out = output_dir / "promp_library_personalized.npz"
    
    library_data_out = {}
    for task_id, task_data in personalized_library.items():
        library_data_out[f"task_{task_id}_mean"] = task_data['mean']
        library_data_out[f"task_{task_id}_cov"] = task_data['cov']
        library_data_out[f"task_{task_id}_original_mean"] = task_data['original_mean']
    
    library_data_out["tasks"] = np.array(list(personalized_library.keys()))
    library_data_out["n_basis"] = library_data['n_basis']
    library_data_out["width"] = library_data['width']
    library_data_out["ridge"] = library_data['ridge']
    library_data_out["target_user_id"] = np.array([target_user_id])
    library_data_out["style_offset"] = style_offset
    library_data_out["style_component_indices"] = np.array(style_component_indices)
    
    np.savez(library_file_out, **library_data_out)
    
    # Save summary
    summary = {
        "source_library": str(library_file),
        "target_user_id": target_user_id,
        "n_tasks": len(personalized_library),
        "tasks": sorted(personalized_library.keys()),
        "style_components": [i+1 for i in style_component_indices],  # 1-indexed for display
        "style_offset_norm": float(np.linalg.norm(style_offset)),
        "n_target_trajectories": len(target_weights),
        "personalized_library_file": str(library_file_out),
    }
    
    summary_file = output_dir / "personalization_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nPersonalized library saved to: {output_dir}")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Personalize ProMP library using style components.")
    parser.add_argument(
        "--library",
        type=str,
        default="User_1_ProMP_library/promp_library.npz",
        help="Path to source ProMP library.",
    )
    parser.add_argument(
        "--pca",
        type=str,
        default="ProMP/promp_pca_results.npz",
        help="Path to PCA results.",
    )
    parser.add_argument(
        "--anova",
        type=str,
        default="ProMP/style_anova_results.json",
        help="Path to ANOVA results.",
    )
    parser.add_argument(
        "--target_user",
        type=str,
        required=True,
        help="Target user ID to personalize for (e.g., '2').",
    )
    parser.add_argument(
        "--trajectories_dir",
        type=str,
        default="output",
        help="Directory containing target user's trajectory CSVs.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for personalized library (default: User_{target_user}_ProMP_library_personalized).",
    )
    parser.add_argument("--n_basis", type=int, default=20, help="RBF basis count.")
    parser.add_argument("--width", type=float, default=0.05, help="RBF width.")
    parser.add_argument("--ridge", type=float, default=1e-6, help="Ridge regularizer.")
    args = parser.parse_args()
    
    library_file = Path(args.library)
    pca_file = Path(args.pca)
    anova_file = Path(args.anova)
    trajectories_dir = Path(args.trajectories_dir)
    
    if args.out_dir is None:
        out_dir = Path(f"User_{args.target_user}_ProMP_library_personalized")
    else:
        out_dir = Path(args.out_dir)
    
    if not library_file.exists():
        raise FileNotFoundError(f"Library file not found: {library_file}")
    if not pca_file.exists():
        raise FileNotFoundError(f"PCA file not found: {pca_file}")
    if not anova_file.exists():
        raise FileNotFoundError(f"ANOVA file not found: {anova_file}")
    
    personalize_library(
        library_file,
        pca_file,
        anova_file,
        args.target_user,
        out_dir,
        trajectories_dir,
        args.n_basis,
        args.width,
        args.ridge
    )


if __name__ == "__main__":
    main()
