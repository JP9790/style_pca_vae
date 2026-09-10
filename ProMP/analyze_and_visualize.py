#!/usr/bin/env python3
"""
Summarize tasks shared by users and visualize PCA results.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Set matplotlib backend before importing pyplot
import os
_root_dir = Path(__file__).resolve().parent
_mpl_cache = _root_dir / ".mplcache"
_cache_home = _root_dir / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_home))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


def summarize_tasks(pca_file: Path):
    """Summarize which tasks are shared by users."""
    data = np.load(pca_file)
    tasks = data['tasks']
    users = data['users']
    
    print("=" * 60)
    print("TASK SUMMARY BY USER")
    print("=" * 60)
    print(f"Total trajectories: {len(tasks)}")
    print()
    
    # Count tasks per user
    user_task_counts = {}
    for user in np.unique(users):
        user_tasks = tasks[users == user]
        unique_tasks = sorted(set(user_tasks))
        user_task_counts[user] = {
            'total': len(user_tasks),
            'unique': unique_tasks,
            'set': set(user_tasks)
        }
        print(f"User {user}:")
        print(f"  - Total trajectories: {user_task_counts[user]['total']}")
        print(f"  - Unique tasks: {len(unique_tasks)}")
        print(f"  - Tasks: {unique_tasks}")
        print()
    
    # Find shared tasks
    if len(user_task_counts) >= 2:
        user_list = sorted(user_task_counts.keys())
        shared_sets = [user_task_counts[u]['set'] for u in user_list]
        shared_tasks = set.intersection(*shared_sets)
        
        print("=" * 60)
        print("SHARED TASKS")
        print("=" * 60)
        print(f"Tasks shared by all {len(user_list)} users: {len(shared_tasks)}")
        print(f"Shared tasks: {sorted(shared_tasks)}")
        print()
        
        # Tasks unique to each user
        for i, user in enumerate(user_list):
            other_users = [u for u in user_list if u != user]
            other_tasks = set.union(*[user_task_counts[u]['set'] for u in other_users])
            user_only = user_task_counts[user]['set'] - other_tasks
            if user_only:
                print(f"Tasks unique to User {user}: {sorted(user_only)}")
    
    return user_task_counts


def visualize_pca(pca_file: Path, output_file: Path):
    """Visualize PCA results with task and user information."""
    data = np.load(pca_file)
    scores = data['scores']
    tasks = data['tasks']
    users = data['users']
    eigvals = data['eigvals']
    
    # Calculate explained variance
    explained_var = eigvals / np.sum(eigvals)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: PC1 vs PC2 colored by task
    ax1 = plt.subplot(2, 3, 1)
    unique_tasks = sorted(set(tasks.tolist()))
    colors_task = plt.cm.tab20(np.linspace(0, 1, len(unique_tasks)))
    for idx, task in enumerate(unique_tasks):
        mask = tasks == task
        ax1.scatter(
            scores[mask, 0],
            scores[mask, 1],
            s=50,
            alpha=0.7,
            label=f'task_{task}',
            color=colors_task[idx],
        )
    ax1.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', fontsize=10)
    ax1.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', fontsize=10)
    ax1.set_title('PCA: PC1 vs PC2 (colored by Task)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    if len(unique_tasks) <= 15:
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
    
    # Plot 2: PC1 vs PC2 colored by user
    ax2 = plt.subplot(2, 3, 2)
    unique_users = sorted(set(users.tolist()))
    colors_user = plt.cm.Set1(np.linspace(0, 1, len(unique_users)))
    for idx, user in enumerate(unique_users):
        mask = users == user
        ax2.scatter(
            scores[mask, 0],
            scores[mask, 1],
            s=50,
            alpha=0.7,
            label=f'User {user}',
            color=colors_user[idx],
        )
    ax2.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', fontsize=10)
    ax2.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', fontsize=10)
    ax2.set_title('PCA: PC1 vs PC2 (colored by User)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # Plot 3: PC2 vs PC3
    if scores.shape[1] >= 3:
        ax3 = plt.subplot(2, 3, 3)
        for idx, user in enumerate(unique_users):
            mask = users == user
            ax3.scatter(
                scores[mask, 1],
                scores[mask, 2],
                s=50,
                alpha=0.7,
                label=f'User {user}',
                color=colors_user[idx],
            )
        ax3.set_xlabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', fontsize=10)
        ax3.set_ylabel(f'PC3 ({explained_var[2]*100:.1f}% variance)', fontsize=10)
        ax3.set_title('PCA: PC2 vs PC3 (colored by User)', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)
    
    # Plot 4: Explained variance
    ax4 = plt.subplot(2, 3, 4)
    n_components_show = min(20, len(explained_var))
    ax4.bar(range(1, n_components_show + 1), explained_var[:n_components_show] * 100)
    ax4.set_xlabel('Principal Component', fontsize=10)
    ax4.set_ylabel('Explained Variance (%)', fontsize=10)
    ax4.set_title('Explained Variance by Component', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Cumulative explained variance
    ax5 = plt.subplot(2, 3, 5)
    cumulative_var = np.cumsum(explained_var) * 100
    n_components_show = min(20, len(cumulative_var))
    ax5.plot(range(1, n_components_show + 1), cumulative_var[:n_components_show], 'o-', linewidth=2, markersize=6)
    ax5.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='80% threshold')
    ax5.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% threshold')
    ax5.set_xlabel('Number of Components', fontsize=10)
    ax5.set_ylabel('Cumulative Explained Variance (%)', fontsize=10)
    ax5.set_title('Cumulative Explained Variance', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=8)
    
    # Plot 6: PC scores distribution by user
    ax6 = plt.subplot(2, 3, 6)
    for idx, user in enumerate(unique_users):
        mask = users == user
        ax6.hist(scores[mask, 0], bins=15, alpha=0.6, label=f'User {user}', 
                color=colors_user[idx], edgecolor='black')
    ax6.set_xlabel(f'PC1 Score', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.set_title('PC1 Score Distribution by User', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze tasks and visualize PCA results.")
    parser.add_argument(
        "--pca",
        type=str,
        default="ProMP/promp_pca_results.npz",
        help="Path to PCA results .npz.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="ProMP/vis/promp_pca_analysis.png",
        help="Output visualization PNG path.",
    )
    args = parser.parse_args()
    
    pca_file = Path(args.pca)
    if not pca_file.exists():
        raise FileNotFoundError(f"PCA results file not found: {pca_file}")
    
    # Summarize tasks
    user_task_counts = summarize_tasks(pca_file)
    
    # Visualize PCA
    output_file = Path(args.out)
    visualize_pca(pca_file, output_file)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
