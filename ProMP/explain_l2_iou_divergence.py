"""
Explanation script for why L2-norm and IoU can show opposite trends.

This script visualizes examples where L2-norm and IoU metrics diverge,
helping understand the fundamental differences between these metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.spatial.distance import cdist


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


def analyze_metrics_divergence():
    """Analyze why L2-norm and IoU show opposite trends."""
    
    # Load the actual data
    alphabet_l2_file = Path("ProMP/vis/alphabet_library_l2_norm_statistics.csv")
    alphabet_iou_file = Path("ProMP/vis/alphabet_library_iou_statistics.csv")
    
    if not alphabet_l2_file.exists() or not alphabet_iou_file.exists():
        print("CSV files not found. Please run the personalization scripts first.")
        return
    
    l2_df = pd.read_csv(alphabet_l2_file)
    iou_df = pd.read_csv(alphabet_iou_file)
    
    # Merge dataframes
    merged = pd.merge(l2_df, iou_df, on='Task', suffixes=('_l2', '_iou'))
    
    # Calculate improvements
    merged['L2_improvement'] = merged['Generic_Mean_l2'] - merged['Personalized_Mean_l2']  # Positive = improvement
    merged['IoU_improvement'] = merged['Personalized_Mean_iou'] - merged['Generic_Mean_iou']  # Positive = improvement
    
    # Find tasks with opposite trends
    opposite_trends = merged[
        (merged['L2_improvement'] > 0) & (merged['IoU_improvement'] < 0) |  # L2 improves, IoU worsens
        (merged['L2_improvement'] < 0) & (merged['IoU_improvement'] > 0)    # L2 worsens, IoU improves
    ]
    
    print("=" * 80)
    print("ANALYSIS: Why L2-norm and IoU Show Opposite Trends")
    print("=" * 80)
    print("\n1. FUNDAMENTAL DIFFERENCES:")
    print("-" * 80)
    print("L2-norm (Euclidean Distance):")
    print("  • Measures: Sum of squared point-to-point distances")
    print("  • Sensitivity: ALL distances contribute (no threshold)")
    print("  • Lower is better: Smaller total distance = better match")
    print("  • Affected by: Systematic offsets, scale differences, all point deviations")
    print()
    print("IoU (Intersection over Union):")
    print("  • Measures: Fraction of points within threshold distance (default: 5.0)")
    print("  • Sensitivity: Binary matching (within/outside threshold)")
    print("  • Higher is better: More points within threshold = better match")
    print("  • Affected by: Local shape similarity, threshold choice, point density")
    print()
    
    print("\n2. SCENARIOS WHERE METRICS DIVERGE:")
    print("-" * 80)
    print("Scenario A: Systematic Offset")
    print("  • Personalization shifts entire trajectory by small constant")
    print("  • L2-norm: Increases (all points shifted)")
    print("  • IoU: May stay same or improve (if shift < threshold)")
    print()
    print("Scenario B: Local Shape Improvement with Global Deviation")
    print("  • Personalization improves local shape matching")
    print("  • But introduces a few large deviations")
    print("  • L2-norm: May improve (many small distances reduced)")
    print("  • IoU: May decrease (fewer points within threshold due to large deviations)")
    print()
    print("Scenario C: Scale/Curvature Changes")
    print("  • Personalization changes trajectory scale or curvature")
    print("  • L2-norm: Sensitive to all point distances")
    print("  • IoU: More forgiving if overall shape is preserved")
    print()
    
    print("\n3. TASKS WITH OPPOSITE TRENDS (Alphabet Library):")
    print("-" * 80)
    if len(opposite_trends) > 0:
        for _, row in opposite_trends.iterrows():
            task = row['Task']
            l2_gen = row['Generic_Mean_l2']
            l2_pers = row['Personalized_Mean_l2']
            iou_gen = row['Generic_Mean_iou']
            iou_pers = row['Personalized_Mean_iou']
            
            print(f"\nTask '{task}':")
            print(f"  L2-norm: {l2_gen:.2f} → {l2_pers:.2f} (Δ = {row['L2_improvement']:.2f})")
            print(f"  IoU:     {iou_gen:.4f} → {iou_pers:.4f} (Δ = {row['IoU_improvement']:.4f})")
            
            if row['L2_improvement'] > 0 and row['IoU_improvement'] < 0:
                print(f"  → L2 improves but IoU decreases")
                print(f"    Possible reason: Better overall point matching, but some regions")
                print(f"    deviate beyond threshold (5.0), reducing IoU")
            else:
                print(f"  → L2 worsens but IoU improves")
                print(f"    Possible reason: Systematic offset or scale change that increases")
                print(f"    point distances, but shape similarity keeps points within threshold")
    else:
        print("No tasks found with opposite trends.")
    
    print("\n4. RECOMMENDATIONS:")
    print("-" * 80)
    print("• Use L2-norm when: You care about exact point-to-point accuracy")
    print("• Use IoU when: You care about spatial overlap and shape similarity")
    print("• Consider both metrics: They provide complementary information")
    print("• Adjust IoU threshold: Lower threshold (e.g., 3.0) = stricter matching")
    print("• Visual inspection: Always check detailed trajectory plots")
    print()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: L2-norm comparison
    ax1 = axes[0, 0]
    tasks = merged['Task'].values
    x_pos = np.arange(len(tasks))
    width = 0.35
    ax1.bar(x_pos - width/2, merged['Generic_Mean_l2'], width, 
            label='Generic', alpha=0.8, color='blue')
    ax1.bar(x_pos + width/2, merged['Personalized_Mean_l2'], width,
            label='Personalized', alpha=0.8, color='red')
    ax1.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax1.set_ylabel('L2-norm', fontsize=12, fontweight='bold')
    ax1.set_title('L2-norm Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(tasks)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: IoU comparison
    ax2 = axes[0, 1]
    ax2.bar(x_pos - width/2, merged['Generic_Mean_iou'], width,
            label='Generic', alpha=0.8, color='blue')
    ax2.bar(x_pos + width/2, merged['Personalized_Mean_iou'], width,
            label='Personalized', alpha=0.8, color='red')
    ax2.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax2.set_ylabel('IoU', fontsize=12, fontweight='bold')
    ax2.set_title('IoU Comparison (Higher is Better)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(tasks)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    # Plot 3: Improvement comparison
    ax3 = axes[1, 0]
    # Normalize improvements for comparison
    l2_improvement_norm = merged['L2_improvement'] / merged['Generic_Mean_l2'] * 100  # Percentage
    iou_improvement_norm = merged['IoU_improvement'] * 100  # Percentage points
    
    ax3.scatter(l2_improvement_norm, iou_improvement_norm, s=100, alpha=0.7, c='purple')
    for i, task in enumerate(tasks):
        ax3.annotate(task, (l2_improvement_norm.iloc[i], iou_improvement_norm.iloc[i]),
                    fontsize=10, ha='center', va='bottom')
    
    ax3.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('L2-norm Improvement (%)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('IoU Improvement (percentage points)', fontsize=12, fontweight='bold')
    ax3.set_title('Improvement Correlation', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Highlight opposite trends
    ax4 = axes[1, 1]
    if len(opposite_trends) > 0:
        opposite_tasks = opposite_trends['Task'].values
        opposite_l2 = opposite_trends['L2_improvement'].values
        opposite_iou = opposite_trends['IoU_improvement'].values
        
        # Create a bar plot showing the divergence
        x_opp = np.arange(len(opposite_tasks))
        ax4_twin = ax4.twinx()
        
        bars1 = ax4.bar(x_opp - 0.2, opposite_l2, 0.4, label='L2 Improvement', 
                       color='green', alpha=0.7)
        bars2 = ax4_twin.bar(x_opp + 0.2, opposite_iou, 0.4, label='IoU Change', 
                            color='orange', alpha=0.7)
        
        ax4.set_xlabel('Task', fontsize=12, fontweight='bold')
        ax4.set_ylabel('L2-norm Improvement', fontsize=12, fontweight='bold', color='green')
        ax4_twin.set_ylabel('IoU Change', fontsize=12, fontweight='bold', color='orange')
        ax4.set_title('Tasks with Opposite Trends', fontsize=14, fontweight='bold')
        ax4.set_xticks(x_opp)
        ax4.set_xticklabels(opposite_tasks)
        ax4.axhline(0, color='black', linestyle='-', linewidth=1)
        ax4_twin.axhline(0, color='black', linestyle='-', linewidth=1)
        ax4.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='best')
    else:
        ax4.text(0.5, 0.5, 'No tasks with opposite trends', 
                ha='center', va='center', fontsize=14, transform=ax4.transAxes)
        ax4.set_title('Tasks with Opposite Trends', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = Path("ProMP/vis/l2_iou_divergence_analysis.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    plt.close()


if __name__ == "__main__":
    analyze_metrics_divergence()
