#!/usr/bin/env python3
"""
Learn ProMP from trajectory CSVs in library_out and save results to generic_library.
Each task has 5 demonstrations, and ProMP results are saved in task_X_initialpromp subfolders.
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


def compute_promp_distribution(weights: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ProMP distribution (mean and covariance) from multiple weight samples.
    
    Args:
        weights: List of weight vectors, each of shape (P,)
        
    Returns:
        mean: Mean weight vector of shape (P,)
        cov: Covariance matrix of shape (P, P)
    """
    weights_arr = np.vstack(weights)  # (N, P)
    mean = np.mean(weights_arr, axis=0)  # (P,)
    
    if len(weights) > 1:
        cov = np.cov(weights_arr.T)  # (P, P)
    else:
        # If only one sample, use a small diagonal covariance
        cov = np.eye(len(mean)) * 1e-6
    
    return mean, cov


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn ProMP from library_out trajectories.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="library_out",
        help="Input folder containing trajectory CSVs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generic_library",
        help="Output directory for ProMP results.",
    )
    parser.add_argument("--n_basis", type=int, default=20, help="RBF basis count.")
    parser.add_argument("--width", type=float, default=0.05, help="RBF width.")
    parser.add_argument("--ridge", type=float, default=1e-6, help="Ridge regularizer.")
    args = parser.parse_args()

    input_root = Path(args.input_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    
    # Load all trajectory CSVs
    print(f"Loading trajectory data from {input_root}...")
    demos: List[Demo] = []
    for csv_path in input_root.rglob("*-trajectory.csv"):
        try:
            time, y = read_trajectory_csv(csv_path)
            if time.size < 2:
                continue
            task_id = infer_task_id_from_path(csv_path)
            demos.append(Demo(time=time, y=y, task=task_id, path=csv_path))
        except Exception as e:
            print(f"Warning: Skipping {csv_path}: {e}")
            continue
    
    if not demos:
        raise SystemExit(f"No trajectory CSVs found under {input_root}")
    
    demos.sort(key=lambda d: (d.task, d.path.as_posix()))
    print(f"Loaded {len(demos)} trajectory files")
    
    # Group demos by task
    task_demos = defaultdict(list)
    for demo in demos:
        task_demos[demo.task].append(demo)
    
    print(f"Found {len(task_demos)} unique tasks")
    
    # Learn ProMP for each task
    print("\nLearning ProMP for each task...")
    for task_id in sorted(task_demos.keys()):
        task_demo_list = task_demos[task_id]
        print(f"  Task {task_id}: {len(task_demo_list)} demonstrations")
        
        if len(task_demo_list) == 0:
            print(f"    Warning: No demonstrations for task {task_id}, skipping")
            continue
        
        # Fit weights for each demonstration
        weights_list = []
        for demo in task_demo_list:
            w = fit_weights(demo.time, demo.y, args.n_basis, args.width, args.ridge)
            weights_list.append(w)
        
        # Compute ProMP distribution (mean and covariance)
        mean, cov = compute_promp_distribution(weights_list)
        
        # Create output subfolder: task_X_initialpromp
        output_subdir = output_root / f"task_{task_id}_initialpromp"
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Save ProMP results
        promp_file = output_subdir / "promp.npz"
        np.savez(
            promp_file,
            mean=mean,
            cov=cov,
            n_demos=len(task_demo_list),
            n_basis=np.array([args.n_basis]),
            width=np.array([args.width]),
            ridge=np.array([args.ridge]),
        )
        
        # Save summary JSON
        summary = {
            "task": task_id,
            "n_demos": len(task_demo_list),
            "n_basis": args.n_basis,
            "width": args.width,
            "ridge": args.ridge,
            "weight_dim": int(mean.shape[0]),
            "promp_file": str(promp_file),
        }
        
        summary_file = output_subdir / "summary.json"
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        print(f"    Saved to: {output_subdir}")
    
    print(f"\nProMP learning complete! Results saved to: {output_root}")
    print(f"Total tasks processed: {len(task_demos)}")


if __name__ == "__main__":
    main()
