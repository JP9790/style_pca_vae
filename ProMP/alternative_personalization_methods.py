#!/usr/bin/env python3
"""
Alternative personalization methods using identified style components.

Beyond the simple style offset approach, this module implements several alternative
strategies for utilizing style components to personalize ProMP libraries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

np.seterr(all="ignore")


def get_style_components(anova_file: Path, style_labels: List[str] = None) -> List[int]:
    """Get style component indices from ANOVA results."""
    if style_labels is None:
        style_labels = ['style', 'style_candidate']
    
    with anova_file.open('r') as f:
        anova_data = json.load(f)
    
    style_components = []
    for result in anova_data['results']:
        if result['label'] in style_labels:
            style_components.append(result['component'] - 1)
    
    return sorted(style_components)


def compute_style_coordinates(
    target_weights: np.ndarray,
    pca_mean: np.ndarray,
    pca_eigvecs: np.ndarray,
    style_component_indices: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute style coordinates for target user's weights.
    
    Returns:
        style_coordinates: (N, d_s) - style coordinates for each weight
        style_mean: (d_s,) - average style coordinates
        V_s: (P, d_s) - style subspace basis
    """
    if len(style_component_indices) == 0:
        return np.zeros((target_weights.shape[0], 0)), np.zeros(0), np.zeros((target_weights.shape[1], 0))
    
    V_s = pca_eigvecs[:, style_component_indices]  # (P, d_s)
    centered_weights = target_weights - pca_mean  # (N, P)
    style_coordinates = centered_weights @ V_s  # (N, d_s)
    style_mean = np.mean(style_coordinates, axis=0)  # (d_s,)
    
    return style_coordinates, style_mean, V_s


# ============================================================================
# METHOD 1: Simple Style Offset (Current Baseline)
# ============================================================================

def method1_style_offset(
    library_means: Dict[str, np.ndarray],
    style_mean: np.ndarray,
    V_s: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Method 1: Simple additive style offset (baseline).
    
    μ_k^styled = μ_k + V_s * z_bar^(s)
    """
    style_offset = V_s @ style_mean
    personalized_means = {}
    for task, mean in library_means.items():
        personalized_means[task] = mean + style_offset
    return personalized_means


# ============================================================================
# METHOD 2: Style Scaling/Modulation
# ============================================================================

def method2_style_scaling(
    library_means: Dict[str, np.ndarray],
    pca_mean: np.ndarray,
    style_mean: np.ndarray,
    V_s: np.ndarray,
    scale_factor: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Method 2: Scale style components by a factor.
    
    This allows controlling the strength of personalization.
    μ_k^styled = μ_k + scale_factor * V_s * z_bar^(s)
    """
    style_offset = scale_factor * (V_s @ style_mean)
    personalized_means = {}
    for task, mean in library_means.items():
        personalized_means[task] = mean + style_offset
    return personalized_means


# ============================================================================
# METHOD 3: Task-Specific Style Modulation
# ============================================================================

def method3_task_specific_style(
    library_means: Dict[str, np.ndarray],
    target_weights: np.ndarray,
    target_tasks: List[str],
    pca_mean: np.ndarray,
    style_mean: np.ndarray,
    V_s: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Method 3: Task-specific style modulation.
    
    For each task, compute style coordinates from that task's weights only,
    then use a weighted combination of global and task-specific style.
    
    μ_k^styled = μ_k + α * V_s * z_bar^(s) + (1-α) * V_s * z_k^(s)
    where z_k^(s) is the style coordinate for task k.
    """
    # Compute task-specific style coordinates
    task_style_coords = {}
    for task in set(target_tasks):
        task_mask = np.array([t == task for t in target_tasks])
        if task_mask.sum() > 0:
            task_weights = target_weights[task_mask]
            task_centered = task_weights - pca_mean
            task_style_coords[task] = np.mean(task_centered @ V_s, axis=0)
        else:
            task_style_coords[task] = style_mean  # Fallback to global
    
    # Blend global and task-specific style (α = 0.5)
    alpha = 0.5
    personalized_means = {}
    for task, mean in library_means.items():
        if task in task_style_coords:
            task_style = task_style_coords[task]
            blended_style = alpha * style_mean + (1 - alpha) * task_style
        else:
            blended_style = style_mean  # Use global if task not in target data
        style_offset = V_s @ blended_style
        personalized_means[task] = mean + style_offset
    
    return personalized_means


# ============================================================================
# METHOD 4: Style in Covariance
# ============================================================================

def method4_style_in_covariance(
    library_means: Dict[str, np.ndarray],
    library_covs: Dict[str, np.ndarray],
    style_mean: np.ndarray,
    V_s: np.ndarray,
    style_cov_scale: float = 0.1
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Method 4: Modify both mean and covariance based on style.
    
    μ_k^styled = μ_k + V_s * z_bar^(s)
    Σ_k^styled = Σ_k + style_cov_scale * V_s * V_s^T
    
    This adds style-dependent variability to the covariance.
    """
    style_offset = V_s @ style_mean
    style_cov_addition = style_cov_scale * (V_s @ V_s.T)
    
    personalized_means = {}
    personalized_covs = {}
    for task, mean in library_means.items():
        personalized_means[task] = mean + style_offset
        personalized_covs[task] = library_covs[task] + style_cov_addition
    
    return personalized_means, personalized_covs


# ============================================================================
# METHOD 5: Style Interpolation
# ============================================================================

def method5_style_interpolation(
    library_means: Dict[str, np.ndarray],
    generic_means: Dict[str, np.ndarray],
    style_mean: np.ndarray,
    V_s: np.ndarray,
    interpolation_weight: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Method 5: Interpolate between generic and fully personalized.
    
    μ_k^styled = (1-α) * μ_k^generic + α * (μ_k^generic + V_s * z_bar^(s))
    where α controls the interpolation strength.
    """
    style_offset = V_s @ style_mean
    personalized_means = {}
    for task, mean in library_means.items():
        if task in generic_means:
            generic_mean = generic_means[task]
            fully_personalized = generic_mean + style_offset
            personalized_means[task] = (1 - interpolation_weight) * generic_mean + \
                                       interpolation_weight * fully_personalized
        else:
            # Fallback to simple offset if generic not available
            personalized_means[task] = mean + style_offset
    return personalized_means


# ============================================================================
# METHOD 6: Style-Weighted Combination
# ============================================================================

def method6_style_weighted_combination(
    library_means: Dict[str, np.ndarray],
    target_weights: np.ndarray,
    target_tasks: List[str],
    pca_mean: np.ndarray,
    style_mean: np.ndarray,
    V_s: np.ndarray,
    weight_threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Method 6: Weighted combination based on style confidence.
    
    For each task, compute how "confident" we are in the style (based on
    variance of style coordinates), then weight the personalization accordingly.
    
    μ_k^styled = μ_k + confidence_k * V_s * z_bar^(s)
    """
    # Compute style coordinate variance per task
    task_style_vars = {}
    for task in set(target_tasks):
        task_mask = np.array([t == task for t in target_tasks])
        if task_mask.sum() > 1:  # Need at least 2 samples for variance
            task_weights = target_weights[task_mask]
            task_centered = task_weights - pca_mean
            task_style_coords = task_centered @ V_s
            task_style_vars[task] = np.var(task_style_coords, axis=0).mean()
        else:
            task_style_vars[task] = 1.0  # High variance if only one sample
    
    # Convert variance to confidence (inverse relationship)
    max_var = max(task_style_vars.values()) if task_style_vars else 1.0
    style_offset = V_s @ style_mean
    
    personalized_means = {}
    for task, mean in library_means.items():
        if task in task_style_vars:
            # Lower variance = higher confidence
            confidence = 1.0 / (1.0 + task_style_vars[task] / max_var)
            confidence = max(weight_threshold, min(1.0, confidence))  # Clamp
        else:
            confidence = 0.5  # Default confidence
        
        personalized_means[task] = mean + confidence * style_offset
    
    return personalized_means


# ============================================================================
# METHOD 7: Style Subspace Projection
# ============================================================================

def method7_style_subspace_projection(
    library_means: Dict[str, np.ndarray],
    pca_mean: np.ndarray,
    pca_eigvecs: np.ndarray,
    style_component_indices: List[int],
    task_component_indices: List[int]
) -> Dict[str, np.ndarray]:
    """
    Method 7: Project ProMP onto style subspace, keep task components separate.
    
    Decompose each task mean into task-related and style-related parts,
    then replace style part with user's style.
    
    μ_k = μ_task + μ_style
    μ_k^styled = μ_task + V_s * z_bar^(s)
    """
    if len(task_component_indices) == 0:
        # If no task components identified, use all non-style components
        all_indices = set(range(pca_eigvecs.shape[1]))
        task_component_indices = sorted(list(all_indices - set(style_component_indices)))
    
    V_s = pca_eigvecs[:, style_component_indices]  # (P, d_s)
    V_t = pca_eigvecs[:, task_component_indices]  # (P, d_t)
    
    personalized_means = {}
    for task, mean in library_means.items():
        # Decompose mean into task and style parts
        centered_mean = mean - pca_mean
        task_part = V_t @ (V_t.T @ centered_mean)
        # Style part will be replaced
        personalized_means[task] = pca_mean + task_part
    
    return personalized_means


# ============================================================================
# METHOD 8: Multi-Component Style (Different weights for different style components)
# ============================================================================

def method8_multi_component_style(
    library_means: Dict[str, np.ndarray],
    style_mean: np.ndarray,
    V_s: np.ndarray,
    component_weights: np.ndarray = None
) -> Dict[str, np.ndarray]:
    """
    Method 8: Weight different style components differently.
    
    Allows emphasizing certain style aspects over others.
    
    μ_k^styled = μ_k + V_s * (W * z_bar^(s))
    where W is a diagonal weight matrix for style components.
    """
    if component_weights is None:
        # Default: equal weights
        component_weights = np.ones(len(style_mean))
    elif len(component_weights) != len(style_mean):
        raise ValueError(f"component_weights length {len(component_weights)} != style_mean length {len(style_mean)}")
    
    weighted_style = component_weights * style_mean
    style_offset = V_s @ weighted_style
    
    personalized_means = {}
    for task, mean in library_means.items():
        personalized_means[task] = mean + style_offset
    
    return personalized_means


# ============================================================================
# Main Function to Compare Methods
# ============================================================================

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


def compare_personalization_methods(
    library_file: Path,
    pca_file: Path,
    anova_file: Path,
    target_user_id: str,
    trajectories_dir: Path,
    output_dir: Path,
    n_basis: int = 20,
    width: float = 0.05,
    ridge: float = 1e-6
):
    """
    Compare different personalization methods and save results.
    """
    
    # Load library
    print("Loading ProMP library...")
    library_data = np.load(library_file)
    library_tasks = library_data['tasks'].astype(str)
    library_means = {task: library_data[f'task_{task}_mean'] for task in library_tasks}
    library_covs = {task: library_data[f'task_{task}_cov'] for task in library_tasks}
    
    # Load PCA
    print("Loading PCA results...")
    pca_data = np.load(pca_file)
    pca_mean = pca_data['mean']
    pca_eigvecs = pca_data['eigvecs']
    
    # Get style and task components
    print("Loading style components...")
    style_indices = get_style_components(anova_file)
    all_indices = set(range(pca_eigvecs.shape[1]))
    task_indices = sorted(list(all_indices - set(style_indices)))
    
    # Load target user trajectories
    print(f"Loading target user {target_user_id} trajectories...")
    target_weights = []
    target_tasks = []
    for csv_path in trajectories_dir.rglob("*-trajectory.csv"):
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
    
    target_weights_arr = np.vstack(target_weights)
    style_coords, style_mean, V_s = compute_style_coordinates(
        target_weights_arr, pca_mean, pca_eigvecs, style_indices
    )
    
    print(f"\nStyle mean: {style_mean}")
    print(f"Style coordinate std: {np.std(style_coords, axis=0)}")
    
    # Apply all methods
    results = {}
    
    print("\n" + "="*60)
    print("APPLYING PERSONALIZATION METHODS")
    print("="*60)
    
    # Method 1: Simple offset (baseline)
    print("\nMethod 1: Simple Style Offset (Baseline)")
    results['method1_offset'] = method1_style_offset(library_means, style_mean, V_s)
    
    # Method 2: Style scaling
    print("Method 2: Style Scaling (0.5x)")
    results['method2_scaling_0.5'] = method2_style_scaling(library_means, pca_mean, style_mean, V_s, 0.5)
    print("Method 2: Style Scaling (1.5x)")
    results['method2_scaling_1.5'] = method2_style_scaling(library_means, pca_mean, style_mean, V_s, 1.5)
    
    # Method 3: Task-specific
    print("Method 3: Task-Specific Style")
    results['method3_task_specific'] = method3_task_specific_style(
        library_means, target_weights_arr, target_tasks, pca_mean, style_mean, V_s
    )
    
    # Method 4: Style in covariance
    print("Method 4: Style in Covariance")
    results['method4_cov'], results['method4_cov_covs'] = method4_style_in_covariance(
        library_means, library_covs, style_mean, V_s, 0.1
    )
    
    # Method 6: Style-weighted
    print("Method 6: Style-Weighted Combination")
    results['method6_weighted'] = method6_style_weighted_combination(
        library_means, target_weights_arr, target_tasks, pca_mean, style_mean, V_s
    )
    
    # Method 8: Multi-component
    print("Method 8: Multi-Component Style (equal weights)")
    results['method8_multi'] = method8_multi_component_style(library_means, style_mean, V_s)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    for method_name, personalized_means in results.items():
        if method_name.endswith('_covs'):
            continue
        
        method_file = output_dir / f"{method_name}_personalized.npz"
        method_data = {
            'tasks': np.array(list(personalized_means.keys())),
            'n_basis': library_data['n_basis'],
            'width': library_data['width'],
            'ridge': library_data['ridge'],
        }
        for task, mean in personalized_means.items():
            method_data[f'task_{task}_mean'] = mean
            method_data[f'task_{task}_cov'] = library_covs[task]
        
        np.savez(method_file, **method_data)
        print(f"  Saved: {method_file}")
    
    # Save comparison summary
    summary = {
        'target_user_id': target_user_id,
        'n_tasks': len(library_means),
        'n_style_components': len(style_indices),
        'style_components': [i+1 for i in style_indices],
        'style_mean': style_mean.tolist(),
        'style_mean_norm': float(np.linalg.norm(style_mean)),
        'methods': list(results.keys()),
    }
    
    summary_file = output_dir / "personalization_methods_comparison.json"
    with summary_file.open('w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nComparison summary saved to: {summary_file}")
    print("\n" + "="*60)
    print("PERSONALIZATION METHODS COMPARISON COMPLETE")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Compare alternative personalization methods.")
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
        default="1",
        help="Target user ID.",
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
        default="ProMP/personalization_methods_comparison",
        help="Output directory for comparison results.",
    )
    parser.add_argument("--n_basis", type=int, default=20, help="RBF basis count.")
    parser.add_argument("--width", type=float, default=0.05, help="RBF width.")
    parser.add_argument("--ridge", type=float, default=1e-6, help="Ridge regularizer.")
    args = parser.parse_args()
    
    compare_personalization_methods(
        Path(args.library),
        Path(args.pca),
        Path(args.anova),
        args.target_user,
        Path(args.trajectories_dir),
        Path(args.out_dir),
        args.n_basis,
        args.width,
        args.ridge
    )


if __name__ == "__main__":
    main()
