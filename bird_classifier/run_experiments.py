#!/usr/bin/env python3
"""
Organized Pseudo-Labeling Experiments for Bird Classification

This script runs multiple pseudo-labeling configurations and compares their performance.
It uses the robust experiment framework added to fewshot.py with:
- Early stopping when accuracy drops 3 consecutive times
- Rollback to best state if final accuracy is worse
- Fresh start for each experiment configuration

Usage:
    Run in the notebook after setting up the experiment:
    
    %run run_experiments.py
    
    Or import and call the function:
    
    from run_experiments import run_all_experiments
    results_df = run_all_experiments(experiment, extractor, ds_train)
"""

import importlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Experiment configurations to test
EXPERIMENT_CONFIGS = [
    # Config 1: Baseline - Mean prototypes with calibrated thresholds
    {
        "name": "Mean+Calibrated",
        "prototype_method": "mean",
        "sim_threshold": 0.40,
        "sim_margin_threshold": 0.04,
        "mutual_nn_k": 0,
        "max_per_class": 5,
    },
    # Config 2: Trimmed Mean - More robust to outliers
    {
        "name": "TrimmedMean+Calibrated", 
        "prototype_method": "trimmed_mean",
        "trim_k": 1,
        "sim_threshold": 0.40,
        "sim_margin_threshold": 0.04,
        "mutual_nn_k": 0,
        "max_per_class": 5,
    },
    # Config 3: Trimmed Mean with aggressive thresholds
    {
        "name": "TrimmedMean+Aggressive",
        "prototype_method": "trimmed_mean",
        "trim_k": 1,
        "sim_threshold": 0.35,
        "sim_margin_threshold": 0.03,
        "mutual_nn_k": 0,
        "max_per_class": 8,
    },
    # Config 4: Weighted prototypes
    {
        "name": "Weighted+Calibrated",
        "prototype_method": "weighted",
        "weight_power": 2.0,
        "sim_threshold": 0.40,
        "sim_margin_threshold": 0.04,
        "mutual_nn_k": 0,
        "max_per_class": 5,
    },
    # Config 5: Trimmed Mean with Mutual NN filtering
    {
        "name": "TrimmedMean+MutualNN(k=150)",
        "prototype_method": "trimmed_mean",
        "trim_k": 1,
        "sim_threshold": 0.38,
        "sim_margin_threshold": 0.035,
        "mutual_nn_k": 150,
        "max_per_class": 6,
    },
    # Config 6: Weighted with Mutual NN filtering
    {
        "name": "Weighted+MutualNN(k=200)",
        "prototype_method": "weighted",
        "weight_power": 2.0,
        "sim_threshold": 0.38,
        "sim_margin_threshold": 0.035,
        "mutual_nn_k": 200,
        "max_per_class": 6,
    },
]


def run_all_experiments(experiment, extractor, ds_train, configs=None, verbose=True):
    """
    Run all pseudo-labeling experiments and return comparison DataFrame.
    
    Args:
        experiment: FewShotExperiment instance (will be reset for each config)
        extractor: FeatureExtractor instance
        ds_train: DeepLake training dataset
        configs: List of experiment configs (defaults to EXPERIMENT_CONFIGS)
        verbose: Whether to print progress
        
    Returns:
        DataFrame with results for each configuration
    """
    # Reload fewshot module to get latest code
    import fewshot
    importlib.reload(fewshot)
    
    if configs is None:
        configs = EXPERIMENT_CONFIGS
    
    # Import the experiment function
    from fewshot import run_pseudo_labeling_experiments
    
    # Run experiments
    results_df = run_pseudo_labeling_experiments(
        experiment=experiment,
        extractor=extractor,
        ds_train=ds_train,
        configs=configs,
        max_iterations=50,
        max_accuracy_drops=3,
        patience=5,
        verbose=verbose
    )
    
    return results_df


def visualize_results(results_df, save_path=None):
    """
    Create visualization of experiment results.
    
    Args:
        results_df: DataFrame from run_all_experiments
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Sort by final accuracy for consistent ordering
    results_sorted = results_df.sort_values('final_accuracy', ascending=False)
    
    # Plot 1: Final Accuracy Comparison
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_sorted)))
    bars = ax1.barh(results_sorted['config_name'], results_sorted['final_accuracy'] * 100, 
                     color=colors, edgecolor='black')
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_title('Final Validation Accuracy by Configuration')
    ax1.set_xlim(45, 60)  # Adjust based on expected range
    
    # Add value labels
    for bar, acc in zip(bars, results_sorted['final_accuracy']):
        ax1.text(acc * 100 + 0.3, bar.get_y() + bar.get_height()/2, 
                f'{acc*100:.2f}%', va='center', fontsize=9)
    
    # Plot 2: Accuracy Improvement
    ax2 = axes[1]
    improvements = results_sorted['final_accuracy'] - results_sorted['initial_accuracy']
    colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
    bars2 = ax2.barh(results_sorted['config_name'], improvements * 100, 
                      color=colors_imp, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Accuracy Change (%)')
    ax2.set_title('Accuracy Improvement from Pseudo-Labeling')
    
    # Add value labels
    for bar, imp in zip(bars2, improvements):
        sign = '+' if imp > 0 else ''
        ax2.text(imp * 100 + (0.1 if imp >= 0 else -0.3), 
                bar.get_y() + bar.get_height()/2, 
                f'{sign}{imp*100:.2f}%', va='center', fontsize=9)
    
    # Plot 3: Samples Added
    ax3 = axes[2]
    ax3.barh(results_sorted['config_name'], results_sorted['samples_added'], 
             color='steelblue', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Samples Added')
    ax3.set_title('Number of Pseudo-Labeled Samples Added')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    best_idx = results_df['final_accuracy'].idxmax()
    best_config = results_df.loc[best_idx]
    print(f"\n🏆 Best Configuration: {best_config['config_name']}")
    print(f"   Final Accuracy: {best_config['final_accuracy']*100:.2f}%")
    print(f"   Improvement: {(best_config['final_accuracy'] - best_config['initial_accuracy'])*100:+.2f}%")
    print(f"   Samples Added: {best_config['samples_added']:.0f}")
    print(f"   Iterations: {best_config['iterations']:.0f}")
    
    return best_config


def run_final_experiment(experiment, extractor, ds_train, best_config, seed=42):
    """
    Run the final experiment with the best configuration.
    
    Args:
        experiment: FewShotExperiment instance
        extractor: FeatureExtractor instance
        ds_train: DeepLake training dataset
        best_config: Dict or Series with best configuration
        seed: Random seed for reproducibility
        
    Returns:
        The experiment instance after running
    """
    import fewshot
    importlib.reload(fewshot)
    from fewshot import run_robust_pseudo_labeling
    
    # Extract config parameters
    if isinstance(best_config, pd.Series):
        config_name = best_config['config_name']
        # Find matching config from EXPERIMENT_CONFIGS
        for cfg in EXPERIMENT_CONFIGS:
            if cfg['name'] == config_name:
                config = cfg.copy()
                break
        else:
            raise ValueError(f"Config '{config_name}' not found in EXPERIMENT_CONFIGS")
    else:
        config = best_config.copy()
    
    print(f"\n{'='*70}")
    print(f"RUNNING FINAL EXPERIMENT: {config['name']}")
    print(f"{'='*70}\n")
    
    # Run robust pseudo-labeling
    result = run_robust_pseudo_labeling(
        experiment=experiment,
        extractor=extractor,
        ds_train=ds_train,
        config=config,
        max_iterations=50,
        max_accuracy_drops=3,
        patience=5,
        verbose=True
    )
    
    # Plot progress
    experiment.plot_progress()
    
    # Spot check results
    from fewshot import spot_check_pseudo_labels
    spot_check_pseudo_labels(experiment, n_samples=20, seed=seed)
    
    return result


if __name__ == "__main__":
    print("This script is meant to be imported and run from the notebook.")
    print("\nExample usage:")
    print("  from run_experiments import run_all_experiments, visualize_results")
    print("  results_df = run_all_experiments(experiment, extractor, ds_train)")
    print("  best_config = visualize_results(results_df)")
