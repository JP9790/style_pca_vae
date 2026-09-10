#!/usr/bin/env python3
"""
Focused visualization of style identification results.
"""
from __future__ import annotations

import argparse
from pathlib import Path

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


def visualize_style_focus(pca_file: Path, anova_file: Path, output_file: Path, alpha: float = 0.05):
    """Create focused visualization on style identification."""
    # Load data
    pca_data = np.load(pca_file)
    scores = pca_data['scores']
    eigvals = pca_data['eigvals']
    explained_var = eigvals / np.sum(eigvals) * 100
    
    anova_data = np.loadtxt(anova_file, delimiter=',', skiprows=1, dtype=str)
    components = anova_data[:, 0].astype(int)
    p_task = anova_data[:, 1].astype(float)
    p_user = anova_data[:, 2].astype(float)
    t_p = anova_data[:, 4].astype(float)
    mean = anova_data[:, 5].astype(float)
    labels = anova_data[:, 6]
    
    # Color scheme
    color_map = {
        'task': '#E74C3C',  # Red
        'style': '#3498DB',  # Blue
        'style_candidate': '#F39C12',  # Orange
        'noise': '#95A5A6'  # Gray
    }
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Decision space (p_task vs p_user)
    ax1 = plt.subplot(2, 3, 1)
    for label in np.unique(labels):
        mask = labels == label
        ax1.scatter(p_task[mask], p_user[mask], 
                   c=color_map.get(label, '#95A5A6'), s=100, alpha=0.7,
                   label=label, edgecolors='black', linewidth=1)
    
    # Add decision boundaries
    ax1.axhline(y=alpha, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'α = {alpha}')
    ax1.axvline(x=alpha, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add text annotations for decision regions
    ax1.text(0.01, 0.5, 'Task\nSignificant', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            transform=ax1.transAxes, ha='left')
    ax1.text(0.5, 0.01, 'User\nSignificant', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            transform=ax1.transAxes, ha='left', va='bottom')
    
    ax1.set_xlabel('P-value (Task Factor)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('P-value (User Factor)', fontsize=11, fontweight='bold')
    ax1.set_title('Two-Factor ANOVA Decision Space', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='upper right')
    
    # Plot 2: Component means with t-test significance
    ax2 = plt.subplot(2, 3, 2)
    for label in np.unique(labels):
        mask = labels == label
        ax2.scatter(components[mask], mean[mask], 
                   c=color_map.get(label, '#95A5A6'), s=120, alpha=0.8,
                   label=label, edgecolors='black', linewidth=1.2)
    
    # Highlight style candidates
    style_cand_mask = labels == 'style_candidate'
    if np.any(style_cand_mask):
        ax2.scatter(components[style_cand_mask], mean[style_cand_mask],
                   c=color_map['style_candidate'], s=200, marker='*',
                   edgecolors='black', linewidth=2, zorder=5,
                   label='Style Candidates (highlighted)')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Component Number', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Mean Score', fontsize=11, fontweight='bold')
    ax2.set_title('Component Means (Style Identification)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='best')
    
    # Plot 3: T-test results for style identification
    ax3 = plt.subplot(2, 3, 3)
    style_mask = (labels == 'style') | (labels == 'style_candidate')
    if np.any(style_mask):
        style_colors = [color_map.get(labels[i], '#95A5A6') for i in range(len(labels)) if style_mask[i]]
        bars = ax3.bar(components[style_mask], t_p[style_mask],
                      color=style_colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax3.axhline(y=alpha, color='red', linestyle='--', linewidth=2, 
                   label=f'α = {alpha}', zorder=10)
        
        # Add labels on bars
        for bar, comp, tp in zip(bars, components[style_mask], t_p[style_mask]):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{comp}\n({tp:.3f})', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
    
    ax3.set_xlabel('Component Number', fontsize=11, fontweight='bold')
    ax3.set_ylabel('T-test P-value', fontsize=11, fontweight='bold')
    ax3.set_title('One-Sample T-test for Style Components', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(fontsize=9)
    
    # Plot 4: Explained variance by classification
    ax4 = plt.subplot(2, 3, 4)
    bar_colors = [color_map.get(labels[i], '#95A5A6') for i in range(len(components))]
    bars = ax4.bar(components, explained_var[:len(components)], 
                   color=bar_colors, edgecolor='black', linewidth=0.8, alpha=0.8)
    ax4.set_xlabel('Component Number', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Explained Variance (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Component Importance by Classification', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Classification summary pie chart
    ax5 = plt.subplot(2, 3, 5)
    unique_labels, counts = np.unique(labels, return_counts=True)
    colors_pie = [color_map.get(label, '#95A5A6') for label in unique_labels]
    wedges, texts, autotexts = ax5.pie(counts, labels=unique_labels, colors=colors_pie,
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax5.set_title('Component Classification Distribution', fontsize=12, fontweight='bold')
    
    # Plot 6: Component index colored by classification
    ax6 = plt.subplot(2, 3, 6)
    y_pos = np.arange(len(components))
    for i, (comp, label) in enumerate(zip(components, labels)):
        ax6.barh(i, 1, left=comp-0.5, color=color_map.get(label, '#95A5A6'),
                edgecolor='black', linewidth=0.5, alpha=0.8)
    
    ax6.set_xlabel('Component Number', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Index', fontsize=11, fontweight='bold')
    ax6.set_title('All Components by Classification', fontsize=12, fontweight='bold')
    ax6.set_yticks([])
    ax6.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[label], 
                                     edgecolor='black', label=label) 
                      for label in np.unique(labels)]
    ax6.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.suptitle('Style Identification Results: Two-Factor ANOVA Analysis', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Focused visualization saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Focused visualization of style identification.")
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
        default="ProMP/vis/style_identification_focus.png",
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
    
    visualize_style_focus(pca_file, anova_file, output_file, args.alpha)


if __name__ == "__main__":
    main()
