#!/usr/bin/env python3
"""
Learn ProMP weights from trajectory CSVs and apply PCA in weight space.
Modified to find null task folders across users and extract taskID/userID from folder structure.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

np.seterr(all="ignore")


@dataclass
class Demo:
    time: np.ndarray  # (T,)
    y: np.ndarray  # (T, D)
    task: str
    user: str
    path: Path


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


def find_null_tasks_across_users(output_dir: Path) -> List[str]:
    """
    Find task folders that are null/empty in two or more user folders simultaneously.
    Returns list of task IDs that are empty in at least 2 users.
    """
    # Get all user folders
    user_folders = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("User_")])
    
    if len(user_folders) < 2:
        return []
    
    # Track which tasks are empty for each user
    task_status = defaultdict(dict)  # task_id -> {user_id: is_empty}
    all_tasks = set()
    
    for user_folder in user_folders:
        user_id = user_folder.name.replace("User_", "")
        # Get all task folders for this user
        task_folders = sorted([d for d in user_folder.iterdir() if d.is_dir() and d.name.startswith("task_")])
        
        for task_folder in task_folders:
            task_id = task_folder.name.replace("task_", "")
            all_tasks.add(task_id)
            
            # Check if task folder is empty (no trajectory CSV files)
            csv_files = list(task_folder.glob("*-trajectory.csv"))
            is_empty = len(csv_files) == 0
            task_status[task_id][user_id] = is_empty
    
    # Find tasks that are empty in at least 2 users
    null_tasks = []
    for task_id in sorted(all_tasks):
        empty_count = sum(1 for is_empty in task_status[task_id].values() if is_empty)
        if empty_count >= 2:
            null_tasks.append(task_id)
    
    return null_tasks


def load_demos_with_task_user(output_dir: Path) -> List[Demo]:
    """
    Load all trajectory CSVs and extract task ID and user ID from folder structure.
    """
    demos: List[Demo] = []
    for csv_path in output_dir.rglob("*-trajectory.csv"):
        try:
            time, y = read_trajectory_csv(csv_path)
            if time.size < 2:
                continue
            task_id = infer_task_id_from_path(csv_path)
            user_id = infer_user_id_from_path(csv_path)
            demos.append(Demo(time=time, y=y, task=task_id, user=user_id, path=csv_path))
        except Exception as e:
            print(f"Warning: Skipping {csv_path}: {e}")
            continue
    
    demos.sort(key=lambda d: (d.user, d.task, d.path.as_posix()))
    return demos


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


def compute_pca(weights: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Apply PCA to ProMP weights as described in the paper.
    
    Following the paper:
    - Compute empirical mean w_bar and covariance matrix Sigma
    - Find principal directions {v_j} via eigendecomposition
    - Compute latent coordinates z_i^(j) = v_j^T (w_i - w_bar)
    
    Args:
        weights: Array of shape (N, P) where N is number of samples, P is weight dimension
        
    Returns:
        Dictionary with mean, cov, eigvals, eigvecs, and scores
    """
    # Empirical mean: w_bar
    mean = np.mean(weights, axis=0)  # (P,)
    
    # Center the data: w_i - w_bar
    centered = weights - mean  # (N, P)
    
    # Empirical covariance: Sigma = (1/(N-1)) * (centered^T @ centered)
    # Note: This is equivalent to np.cov(weights.T)
    cov = (centered.T @ centered) / max(weights.shape[0] - 1, 1)  # (P, P)
    
    # Eigendecomposition: Solve Sigma * v_j = lambda_j * v_j
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending order)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]  # Each column is a principal direction v_j
    
    # Compute latent coordinates: z_i^(j) = v_j^T (w_i - w_bar)
    # scores[i, j] = z_i^(j)
    scores = centered @ eigvecs  # (N, P)
    
    return {
        "mean": mean,
        "cov": cov,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ProMP weights + PCA.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Root folder containing trajectory CSVs.",
    )
    parser.add_argument("--n_basis", type=int, default=20, help="RBF basis count.")
    parser.add_argument("--width", type=float, default=0.05, help="RBF width.")
    parser.add_argument("--ridge", type=float, default=1e-6, help="Ridge regularizer.")
    parser.add_argument(
        "--out",
        type=str,
        default="ProMP/promp_pca_results.npz",
        help="Output npz path.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir).expanduser().resolve()
    
    # Step 1: Find task folders that are null/empty in two or more user folders
    print("Finding null task folders across users...")
    null_tasks = find_null_tasks_across_users(output_root)
    if null_tasks:
        print(f"Found {len(null_tasks)} tasks that are empty in 2+ users: {sorted(null_tasks)}")
    else:
        print("No tasks found that are empty in 2+ users simultaneously.")
    
    # Step 2: Load all demos with task and user IDs extracted from folder structure
    print("\nLoading trajectory data...")
    demos = load_demos_with_task_user(output_root)
    if not demos:
        raise SystemExit(f"No trajectory CSVs found under {output_root}")

    print(f"Loaded {len(demos)} trajectory files")
    
    # Step 2.5: Filter to only tasks shared by ALL users
    from collections import defaultdict
    user_task_sets = defaultdict(set)
    for demo in demos:
        user_task_sets[demo.user].add(demo.task)
    
    # Find intersection of all user task sets
    if len(user_task_sets) > 0:
        shared_tasks = set(list(user_task_sets.values())[0])
        for user, task_set in user_task_sets.items():
            shared_tasks = shared_tasks & task_set
        
        print(f"\nFound {len(shared_tasks)} tasks shared by all {len(user_task_sets)} users: {sorted(shared_tasks)}")
        
        # Filter demos to only shared tasks
        original_count = len(demos)
        demos = [d for d in demos if d.task in shared_tasks]
        print(f"Filtered from {original_count} to {len(demos)} trajectories (only shared tasks)")
    else:
        shared_tasks = set()
        print("Warning: No users found, using all tasks")
    
    # Step 3: Learn ProMP weights for each task of each user
    print("\nLearning ProMP weights for each task of each user...")
    weights = []
    tasks = []
    users = []
    paths = []
    
    for demo in demos:
        w = fit_weights(demo.time, demo.y, args.n_basis, args.width, args.ridge)
        weights.append(w)
        tasks.append(demo.task)
        users.append(demo.user)
        paths.append(str(demo.path))

    weights_arr = np.vstack(weights)  # (N, P) where P = d * n_basis
    
    # Step 4: Apply PCA to ProMP weights as described in the paper
    print("\nApplying PCA to ProMP weight space...")
    pca = compute_pca(weights_arr)

    # Save results
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        weights=weights_arr,
        tasks=np.array(tasks),
        users=np.array(users),
        paths=np.array(paths),
        null_tasks=np.array(null_tasks),
        n_basis=np.array([args.n_basis]),
        width=np.array([args.width]),
        ridge=np.array([args.ridge]),
        **pca,
    )

    # Print summary
    explained_var = pca["eigvals"] / np.sum(pca["eigvals"])
    summary = {
        "n_demos": len(demos),
        "weight_dim": int(weights_arr.shape[1]),
        "n_components": len(pca["eigvals"]),
        "tasks": sorted(set(tasks)),
        "users": sorted(set(users)),
        "null_tasks": sorted(null_tasks),
        "explained_variance_ratio_top5": explained_var[:5].tolist(),
        "out": str(out_path),
    }
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
