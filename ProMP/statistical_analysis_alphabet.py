#!/usr/bin/env python3
"""
Statistical Analysis for Alphabet Evaluation:
- Two-way Repeated Measures ANOVA
- Pairwise tests if conditions are met
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
import warnings

warnings.filterwarnings('ignore')


def load_data(csv_file: Path) -> pd.DataFrame:
    """Load the alphabet evaluation data."""
    df = pd.read_csv(csv_file)
    return df


def prepare_data_for_anova(df: pd.DataFrame, metric: str = 'L2') -> pd.DataFrame:
    """
    Prepare data for repeated measures ANOVA.
    
    Args:
        df: DataFrame with columns User, Task, L2_Generic, L2_Personalized, etc.
        metric: 'L2' or 'IoU'
    
    Returns:
        Long-format DataFrame with columns: User, Task, Method, Value
    """
    # Select relevant columns
    if metric == 'L2':
        gen_col = 'L2_Generic'
        pers_col = 'L2_Personalized'
    elif metric == 'IoU':
        gen_col = 'IoU_Generic'
        pers_col = 'IoU_Personalized'
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Create long-format data
    rows = []
    for _, row in df.iterrows():
        user = row['User']
        task = row['Task']
        
        # Generic method
        rows.append({
            'User': user,
            'Task': task,
            'Method': 'Generic',
            'Value': row[gen_col]
        })
        
        # Personalized method
        rows.append({
            'User': user,
            'Task': task,
            'Method': 'Personalized',
            'Value': row[pers_col]
        })
    
    return pd.DataFrame(rows)


def perform_anova(df_long: pd.DataFrame) -> dict:
    """
    Perform two-way repeated measures ANOVA.
    
    Within-subject factors: Method, Task
    Subject factor: User
    
    Note: Uses linear mixed-effects model approach for unbalanced data.
    """
    print("\n" + "="*80)
    print("TWO-WAY REPEATED MEASURES ANOVA")
    print("="*80)
    print("Within-subject factors: Method (2 levels), Task (multiple levels)")
    print("Subject factor: User")
    print("="*80)
    print("Note: Using mixed-effects approach for unbalanced data")
    print("="*80)
    
    # Try statsmodels AnovaRM first (works for balanced data)
    try:
        # Check if data is balanced
        pivot_check = df_long.pivot_table(
            index=['User', 'Task'],
            columns='Method',
            values='Value',
            aggfunc='mean'
        )
        
        # If we can pivot without issues, try AnovaRM
        anova_model = AnovaRM(
            data=df_long,
            depvar='Value',
            subject='User',
            within=['Method', 'Task'],
            aggregate_func='mean'
        )
        anova_results = anova_model.fit()
        
        print("\nANOVA Results (AnovaRM):")
        print(anova_results.summary())
        
        # Extract p-values
        p_method = anova_results.anova_table.loc['Method', 'Pr > F']
        p_task = anova_results.anova_table.loc['Task', 'Pr > F']
        p_interaction = anova_results.anova_table.loc['Method:Task', 'Pr > F']
        
        results = {
            'p_method': p_method,
            'p_task': p_task,
            'p_interaction': p_interaction,
            'anova_table': anova_results.anova_table,
            'model': anova_results,
            'method': 'AnovaRM'
        }
        
        print(f"\nP-values:")
        print(f"  Method main effect: p = {p_method:.6f}")
        print(f"  Task main effect: p = {p_task:.6f}")
        print(f"  Method × Task interaction: p = {p_interaction:.6f}")
        
        return results
        
    except Exception as e:
        print(f"AnovaRM failed (likely unbalanced data): {e}")
        print("\nUsing alternative mixed-effects approach...")
        
        # Alternative: Use mixed-effects model approach
        return perform_mixed_effects_anova(df_long)


def perform_mixed_effects_anova(df_long: pd.DataFrame) -> dict:
    """
    Perform ANOVA using mixed-effects model approach for unbalanced data.
    Uses linear mixed model with User as random effect.
    """
    try:
        from statsmodels.formula.api import mixedlm
        from statsmodels.stats.anova import anova_lm
        from statsmodels.regression.linear_model import OLS
        import statsmodels.api as sm
        
        # Create dummy variables
        df_analysis = df_long.copy()
        df_analysis['Method_Generic'] = (df_analysis['Method'] == 'Generic').astype(int)
        df_analysis['Method_Personalized'] = (df_analysis['Method'] == 'Personalized').astype(int)
        
        # Create interaction terms
        for task in df_analysis['Task'].unique():
            df_analysis[f'Task_{task}'] = (df_analysis['Task'] == task).astype(int)
            df_analysis[f'Method_Task_{task}'] = df_analysis['Method_Personalized'] * df_analysis[f'Task_{task}']
        
        # Build formula for mixed model
        # Fixed effects: Method, Task, Method:Task
        # Random effect: User
        
        # Simplified approach: Use OLS with User as fixed effect (treating as repeated measures)
        # This approximates repeated measures ANOVA
        
        formula = 'Value ~ C(Method) + C(Task) + C(Method):C(Task) + C(User)'
        
        model = OLS.from_formula(formula, data=df_analysis).fit()
        
        # Get ANOVA table
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        print("\nANOVA Results (Mixed-Effects Approach):")
        print(anova_table)
        
        # Extract p-values
        p_method = anova_table.loc['C(Method)', 'PR(>F)']
        p_task = anova_table.loc['C(Task)', 'PR(>F)']
        p_interaction = anova_table.loc['C(Method):C(Task)', 'PR(>F)']
        
        results = {
            'p_method': p_method,
            'p_task': p_task,
            'p_interaction': p_interaction,
            'anova_table': anova_table,
            'model': model,
            'method': 'Mixed-Effects'
        }
        
        print(f"\nP-values:")
        print(f"  Method main effect: p = {p_method:.6f}")
        print(f"  Task main effect: p = {p_task:.6f}")
        print(f"  Method × Task interaction: p = {p_interaction:.6f}")
        
        return results
        
    except Exception as e:
        print(f"Mixed-effects approach failed: {e}")
        print("\nUsing simplified paired comparison approach...")
        
        # Fallback: Simplified approach
        return perform_simplified_anova(df_long)


def perform_simplified_anova(df_long: pd.DataFrame) -> dict:
    """
    Simplified ANOVA using paired comparisons.
    For unbalanced data, this provides approximate p-values.
    """
    print("\nPerforming simplified ANOVA using paired comparisons...")
    
    # Reshape to wide format
    df_wide = df_long.pivot_table(
        index=['User', 'Task'],
        columns='Method',
        values='Value',
        aggfunc='mean'
    ).reset_index()
    
    # Calculate differences
    df_wide['Difference'] = df_wide['Personalized'] - df_wide['Generic']
    
    # 1. Method main effect: Overall paired t-test across users
    overall_diff = df_wide.groupby('User')['Difference'].mean()
    t_stat_method, p_method = stats.ttest_1samp(overall_diff, 0)
    
    # 2. Task main effect: One-way ANOVA on Generic values across tasks
    generic_by_task = df_wide.groupby('Task')['Generic'].apply(list)
    task_arrays = [np.array(vals) for vals in generic_by_task.values if len(vals) > 1]
    if len(task_arrays) > 1:
        f_stat_task, p_task = stats.f_oneway(*task_arrays)
    else:
        p_task = None
    
    # 3. Interaction: Check if differences vary significantly across tasks
    task_diffs = df_wide.groupby('Task')['Difference'].apply(list)
    task_diff_arrays = [np.array(diffs) for diffs in task_diffs.values if len(diffs) > 1]
    if len(task_diff_arrays) > 1:
        f_stat_interaction, p_interaction = stats.f_oneway(*task_diff_arrays)
    else:
        p_interaction = None
    
    results = {
        'p_method': p_method,
        'p_task': p_task,
        'p_interaction': p_interaction,
        'anova_table': None,
        'model': None,
        'method': 'Simplified'
    }
    
    print(f"\nP-values (simplified method):")
    print(f"  Method main effect: p = {p_method:.6f}")
    if p_task is not None:
        print(f"  Task main effect: p = {p_task:.6f}")
    else:
        print(f"  Task main effect: Could not compute")
    if p_interaction is not None:
        print(f"  Method × Task interaction: p = {p_interaction:.6f}")
    else:
        print(f"  Method × Task interaction: Could not compute")
    
    return results


def perform_pairwise_tests(df: pd.DataFrame, metric: str = 'L2', 
                          correction_method: str = 'fdr') -> dict:
    """
    Perform pairwise tests after ANOVA.
    
    Args:
        df: Original wide-format DataFrame
        metric: 'L2' or 'IoU'
        correction_method: 'fdr' or 'bonferroni'
    """
    print("\n" + "="*80)
    print("PAIRWISE TESTS")
    print("="*80)
    
    if metric == 'L2':
        gen_col = 'L2_Generic'
        pers_col = 'L2_Personalized'
    elif metric == 'IoU':
        gen_col = 'IoU_Generic'
        pers_col = 'IoU_Personalized'
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # 1. Overall comparison: Paired t-test across users
    # Δ_u = mean_t (Personalized - Generic) for each user
    user_diffs = []
    for user_id in sorted(df['User'].unique()):
        user_df = df[df['User'] == user_id]
        diff = (user_df[pers_col] - user_df[gen_col]).mean()
        user_diffs.append(diff)
    
    user_diffs = np.array(user_diffs)
    
    # Paired t-test
    t_stat_overall, p_overall = stats.ttest_1samp(user_diffs, 0)
    mean_diff = np.mean(user_diffs)
    std_diff = np.std(user_diffs, ddof=1)
    se_diff = std_diff / np.sqrt(len(user_diffs))
    ci_lower = mean_diff - 1.96 * se_diff
    ci_upper = mean_diff + 1.96 * se_diff
    
    print(f"\n1. Overall Comparison (across all tasks):")
    print(f"   Mean difference (Personalized - Generic): {mean_diff:.6f}")
    print(f"   Standard deviation: {std_diff:.6f}")
    print(f"   Standard error: {se_diff:.6f}")
    print(f"   95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"   t-statistic: {t_stat_overall:.4f}")
    print(f"   p-value: {p_overall:.6f}")
    print(f"   Significant: {'Yes' if p_overall < 0.05 else 'No'} (α = 0.05)")
    
    # 2. Per-task comparison: Paired t-test for each task
    print(f"\n2. Per-Task Comparison:")
    print(f"   Multiple comparison correction: {correction_method.upper()}")
    print("-" * 80)
    
    task_results = []
    for task_id in sorted(df['Task'].unique()):
        task_df = df[df['Task'] == task_id]
        
        # Calculate differences for each user
        diffs = task_df[pers_col] - task_df[gen_col]
        
        if len(diffs) < 2:
            continue
        
        # Paired t-test
        t_stat, p_value = stats.ttest_1samp(diffs, 0)
        mean_diff_task = diffs.mean()
        std_diff_task = diffs.std(ddof=1)
        
        task_results.append({
            'Task': task_id,
            'Mean_Difference': mean_diff_task,
            'Std_Difference': std_diff_task,
            't_statistic': t_stat,
            'p_value': p_value,
            'N': len(diffs)
        })
    
    task_results_df = pd.DataFrame(task_results)
    
    # Apply multiple comparison correction
    if correction_method.lower() == 'fdr':
        _, p_corrected, _, _ = multipletests(
            task_results_df['p_value'],
            alpha=0.05,
            method='fdr_bh'
        )
    elif correction_method.lower() == 'bonferroni':
        _, p_corrected, _, _ = multipletests(
            task_results_df['p_value'],
            alpha=0.05,
            method='bonferroni'
        )
    else:
        raise ValueError(f"Unknown correction method: {correction_method}")
    
    task_results_df['p_value_corrected'] = p_corrected
    task_results_df['Significant'] = task_results_df['p_value_corrected'] < 0.05
    
    # Print results
    print(f"{'Task':<8} {'Mean Diff':<12} {'t-stat':<10} {'p-value':<12} {'p-corrected':<15} {'Significant':<12}")
    print("-" * 80)
    for _, row in task_results_df.iterrows():
        print(f"{row['Task']:<8} {row['Mean_Difference']:>11.6f} {row['t_statistic']:>9.4f} "
              f"{row['p_value']:>11.6f} {row['p_value_corrected']:>14.6f} {str(row['Significant']):<12}")
    
    return {
        'overall': {
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'se_diff': se_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            't_statistic': t_stat_overall,
            'p_value': p_overall
        },
        'per_task': task_results_df
    }


def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis for alphabet evaluation data."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="ProMP/all_users_alphabet_evaluation.csv",
        help="Input CSV file with evaluation data"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="L2",
        choices=['L2', 'IoU'],
        help="Metric to analyze: L2 or IoU"
    )
    parser.add_argument(
        "--correction",
        type=str,
        default="fdr",
        choices=['fdr', 'bonferroni'],
        help="Multiple comparison correction method"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ProMP",
        help="Output directory for results"
    )
    args = parser.parse_args()
    
    base_dir = Path(".").resolve()
    input_csv = base_dir / args.input_csv
    output_dir = base_dir / args.output_dir
    
    print("="*80)
    print("STATISTICAL ANALYSIS: ALPHABET EVALUATION")
    print("="*80)
    print(f"Input file: {input_csv}")
    print(f"Metric: {args.metric}")
    print(f"Correction method: {args.correction}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df = load_data(input_csv)
    print(f"Loaded {len(df)} rows")
    print(f"Users: {df['User'].nunique()}")
    print(f"Tasks: {df['Task'].nunique()}")
    
    # Prepare data for ANOVA
    print("\nPreparing data for ANOVA...")
    df_long = prepare_data_for_anova(df, metric=args.metric)
    print(f"Long-format data: {len(df_long)} rows")
    
    # Perform ANOVA
    anova_results = perform_anova(df_long)
    
    # Check conditions
    p_method = anova_results['p_method']
    p_interaction = anova_results['p_interaction']
    
    method_significant = p_method < 0.05 if p_method is not None else False
    interaction_not_significant = p_interaction >= 0.05 if p_interaction is not None else False
    
    print("\n" + "="*80)
    print("ANOVA INTERPRETATION")
    print("="*80)
    print(f"Method significant: {method_significant} (p = {p_method:.6f})")
    print(f"Interaction not significant: {interaction_not_significant} (p = {p_interaction:.6f})")
    
    if method_significant and interaction_not_significant:
        print("\n✓ Conditions met: Method is significant and interaction is not significant")
        print("  → Personalized method improves performance consistently across tasks")
        print("\nProceeding with pairwise tests...")
        
        # Perform pairwise tests
        pairwise_results = perform_pairwise_tests(df, metric=args.metric, 
                                                  correction_method=args.correction)
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save per-task results
        per_task_file = output_dir / f"alphabet_{args.metric.lower()}_per_task_results.csv"
        pairwise_results['per_task'].to_csv(per_task_file, index=False)
        print(f"\nPer-task results saved to: {per_task_file}")
        
        # Save overall summary
        summary_file = output_dir / f"alphabet_{args.metric.lower()}_statistical_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("STATISTICAL ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Metric: {args.metric}\n")
            f.write(f"Correction method: {args.correction}\n\n")
            
            f.write("ANOVA RESULTS\n")
            f.write("-"*80 + "\n")
            f.write(f"Method main effect: p = {p_method:.6f}\n")
            f.write(f"Task main effect: p = {anova_results['p_task']:.6f}\n")
            f.write(f"Method × Task interaction: p = {p_interaction:.6f}\n\n")
            
            f.write("OVERALL COMPARISON\n")
            f.write("-"*80 + "\n")
            overall = pairwise_results['overall']
            f.write(f"Mean difference: {overall['mean_diff']:.6f}\n")
            f.write(f"95% CI: [{overall['ci_lower']:.6f}, {overall['ci_upper']:.6f}]\n")
            f.write(f"t-statistic: {overall['t_statistic']:.4f}\n")
            f.write(f"p-value: {overall['p_value']:.6f}\n")
            f.write(f"Significant: {'Yes' if overall['p_value'] < 0.05 else 'No'}\n\n")
            
            f.write("PER-TASK RESULTS\n")
            f.write("-"*80 + "\n")
            f.write(pairwise_results['per_task'].to_string(index=False))
        
        print(f"Summary saved to: {summary_file}")
        
    else:
        print("\n✗ Conditions not met:")
        if not method_significant:
            print("  - Method is not significant")
        if not interaction_not_significant:
            print("  - Interaction is significant (method effect varies across tasks)")
        print("\nPairwise tests not performed.")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
