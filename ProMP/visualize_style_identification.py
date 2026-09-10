#!/usr/bin/env python3
"""
Visualize the style identification results from two-factor ANOVA analysis.
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
import matplotlib.patches as mpatches


def load_data(pca_file: Path, anova_file: Path):
    """Load PCA and ANOVA results."""
    pca_data = np.load(pca_file)
    scores = pca_data['scores']
    tasks = pca_data['tasks'].astype(str)
    
    if 'users' in pca_data:
        users = pca_data['users'].astype(str)
    else:
        paths = pca_data['paths'].astype(str)
        users = np.array([Path(p).parts[Path(p).parts.index('User_')].replace('User_', '') 
                         if 'User_' in Path(p).parts else 'unknown' 
                         for p in paths])
    
    # Load ANOVA results
    anova_data = np.loadtxt(anova_file, delimiter=',', skiprows=1, dtype=str)
    components = anova_data[:, 0].astype(int)
    p_task = anova_data[:, 1].astype(float)
    p_user = anova_data[:, 2].astype(float)
    t_p = anova_data[:, 4].astype(float)
    mean = anova_data[:, 5].astype(float)
    labels = anova_data[:, 6]
    
    return scores, tasks, users, components, p_task, p_user, t_p, mean, labels


def visualize_style_identification(
    pca_file: Path,
    anova_file: Path,
    output_file: Path,
    alpha: float = 0.05
):
    """Create comprehensive visualization of style identification results."""
    scores, tasks, users, components, p_task, p_user, t_p, mean, labels = load_data(
        pca_file, anova_file
    )
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Color scheme
    color_map = {
        'task': '#FF6B6B',  # Red
        'style': '#4ECDC4',  # Teal
        'style_candidate': '#FFE66D',  # Yellow
        'noise': '#95A5A6'  # Gray
    }
    
    # Plot 1: Component classification bar chart
    ax1 = plt.subplot(3, 3, 1)
    unique_labels, counts = np.unique(labels, return_counts=True)
    colors_bar = [color_map.get(label, '#95A5A6') for label in unique_labels]
    bars = ax1.bar(unique_labels, counts, color=colors_bar, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Number of Components', fontsize=10, fontweight='bold')
    ax1.set_title('Component Classification Summary', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: P-values for task factor
    ax2 = plt.subplot(3, 3, 2)
    colors_p = [color_map.get(label, '#95A5A6') for label in labels]
    ax2.scatter(components, p_task, c=colors_p, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.axhline(y=alpha, color='red', linestyle='--', linewidth=2, label=f'α = {alpha}')
    ax2.set_xlabel('Component', fontsize=10)
    ax2.set_ylabel('P-value (Task)', fontsize=10)
    ax2.set_title('Task Factor Significance', fontsize=11, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    
    # Plot 3: P-values for user factor
    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(components, p_user, c=colors_p, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax3.axhline(y=alpha, color='red', linestyle='--', linewidth=2, label=f'α = {alpha}')
    ax3.set_xlabel('Component', fontsize=10)
    ax3.set_ylabel('P-value (User)', fontsize=10)
    ax3.set_title('User Factor Significance', fontsize=11, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    
    # Plot 4: P-value scatter (task vs user)
    ax4 = plt.subplot(3, 3, 4)
    for label in unique_labels:
        mask = labels == label
        ax4.scatter(p_task[mask], p_user[mask], 
                   c=color_map.get(label, '#95A5A6'), s=80, alpha=0.7,
                   label=label, edgecolors='black', linewidth=0.5)
    ax4.axhline(y=alpha, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax4.axvline(x=alpha, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('P-value (Task)', fontsize=10)
    ax4.set_ylabel('P-value (User)', fontsize=10)
    ax4.set_title('Task vs User P-values', fontsize=11, fontweight='bold')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8, loc='upper right')
    
    # Plot 5: Mean values with t-test significance
    ax5 = plt.subplot(3, 3, 5)
    for label in unique_labels:
        mask = labels == label
        ax5.scatter(components[mask], mean[mask], 
                   c=color_map.get(label, '#95A5A6'), s=80, alpha=0.7,
                   label=label, edgecolors='black', linewidth=0.5)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax5.set_xlabel('Component', fontsize=10)
    ax5.set_ylabel('Mean Score', fontsize=10)
    ax5.set_title('Component Means (for Style Identification)', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=8)
    
    # Plot 6: T-test p-values for style candidates
    ax6 = plt.subplot(3, 3, 6)
    style_mask = (labels == 'style') | (labels == 'style_candidate')
    if np.any(style_mask):
        style_colors = [color_map.get(labels[i], '#95A5A6') for i in range(len(labels)) if style_mask[i]]
        ax6.scatter(components[style_mask], t_p[style_mask], 
                   c=style_colors, s=100, alpha=0.8, edgecolors='black', linewidth=1)
        ax6.axhline(y=alpha, color='red', linestyle='--', linewidth=2, label=f'α = {alpha}')
        ax6.set_xlabel('Component', fontsize=10)
        ax6.set_ylabel('T-test P-value', fontsize=10)
        ax6.set_title('T-test for Style Candidates', fontsize=11, fontweight='bold')
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3)
        ax6.legend(fontsize=8)
    else:
        ax6.text(0.5, 0.5, 'No style candidates found', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('T-test for Style Candidates', fontsize=11, fontweight='bold')
    
    # Plot 7: Component classification by index
    ax7 = plt.subplot(3, 3, 7)
    for label in unique_labels:
        mask = labels == label
        ax7.scatter(components[mask], [1]*np.sum(mask), 
                   c=color_map.get(label, '#95A5A6'), s=200, alpha=0.8,
                   label=label, edgecolors='black', linewidth=1, marker='s')
    ax7.set_xlabel('Component Number', fontsize=10)
    ax7.set_ylabel('', fontsize=10)
    ax7.set_title('Component Classification by Index', fontsize=11, fontweight='bold')
    ax7.set_yticks([])
    ax7.grid(True, alpha=0.3, axis='x')
    ax7.legend(fontsize=8, loc='upper right')
    
    # Plot 8: Component importance by explained variance
    ax8 = plt.subplot(3, 3, 8)
    pca_data = np.load(pca_file)
    eigvals = pca_data['eigvals']
    explained_var = eigvals / np.sum(eigvals) * 100
    
    # Color bars by classification
    bar_colors = [color_map.get(labels[i], '#95A5A6') for i in range(len(components))]
    bars = ax8.bar(components, explained_var[:len(components)], 
                   color=bar_colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    ax8.set_xlabel('Component', fontsize=10)
    ax8.set_ylabel('Explained Variance (%)', fontsize=10)
    ax8.set_title('Component Importance by Classification', fontsize=11, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Plot 9: Summary statistics table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Create summary table
    summary_data = []
    for label in unique_labels:
        mask = labels == label
        n = np.sum(mask)
        if n > 0:
            avg_p_task = np.mean(p_task[mask])
            avg_p_user = np.mean(p_user[mask])
            avg_mean = np.mean(np.abs(mean[mask]))
            summary_data.append([
                label,
                f'{n}',
                f'{avg_p_task:.4f}',
                f'{avg_p_user:.4f}',
                f'{avg_mean:.4f}'
            ])
    
    table = ax9.table(cellText=summary_data,
                      colLabels=['Label', 'Count', 'Avg p_task', 'Avg p_user', 'Avg |mean|'],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color table rows
    for i in range(len(summary_data)):
        color = color_map.get(summary_data[i][0], '#FFFFFF')
        for j in range(5):
            table[(i+1, j)].set_facecolor(color)
            table[(i+1, j)].set_alpha(0.3)
    
    # Header row
    for j in range(5):
        table[(0, j)].set_facecolor('#34495E')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    ax9.set_title('Summary Statistics by Classification', fontsize=11, fontweight='bold', pad=20)
    
    # Add overall title
    fig.suptitle('Style Identification Results from Two-Factor ANOVA', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize style identification results.")
    parser.add_argument(
        "--pca",
        type=str,
        default="ProMP/promp_pca_results.npz",
        help="Path to PCA results .npz.",
    )
    parser.add_argument(
        "--anova",
        type=str,
        default="ProMP/style_anova_results.csv",
        help="Path to ANOVA results CSV.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="ProMP/vis/style_identification.png",
        help="Output visualization PNG path.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level used in analysis.",
    )
    args = parser.parse_args()
    
    pca_file = Path(args.pca)
    anova_file = Path(args.anova)
    output_file = Path(args.out)
    
    if not pca_file.exists():
        raise FileNotFoundError(f"PCA results file not found: {pca_file}")
    if not anova_file.exists():
        raise FileNotFoundError(f"ANOVA results file not found: {anova_file}")
    
    visualize_style_identification(pca_file, anova_file, output_file, args.alpha)


if __name__ == "__main__":
    main()
