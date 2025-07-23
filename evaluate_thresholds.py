#!/usr/bin/env python3
"""
Post-processing script for evaluating anomaly detection with different thresholds.
Loads saved residuals and computes PRC and AUROC curves.
Also evaluates CUSUM algorithm for comparison.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score
import wandb
import yaml
import torch
from src.utils.cusum import CUSUM

def load_residuals(residuals_path):
    """Load residuals from saved file."""
    print(f"Loading residuals from {residuals_path}")
    with open(residuals_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded residuals with {data['dataset_info']['total_samples']} samples")
    print(f"Normal samples: {data['dataset_info']['normal_samples']}")
    print(f"Attack samples: {data['dataset_info']['attack_samples']}")
    
    return data

def evaluate_thresholds(residuals_data, thresholds, temporal_consistency=1):
    """
    Evaluate performance with different thresholds.
    
    Parameters
    ----------
    residuals_data : dict
        Dictionary containing residuals and ground truth
    thresholds : list
        List of threshold values to evaluate
    temporal_consistency : int
        Number of consecutive anomalies required for detection
        
    Returns
    -------
    dict
        Dictionary with evaluation results for each threshold
    """
    y_true = residuals_data['ground_truth']
    scores = residuals_data['error_per_sample_0']  # Use 0-cell errors as anomaly scores
    
    results = {}
    
    for threshold in thresholds:
        # Apply threshold
        raw_predictions = (scores > threshold).astype(int)
        
        # Apply temporal consistency if needed
        if temporal_consistency > 1:
            predictions = apply_temporal_consistency(raw_predictions, temporal_consistency)
        else:
            predictions = raw_predictions
        
        # Calculate metrics
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        
        # Calculate detection rate (percentage of attacks detected)
        attack_indices = np.where(y_true == 1)[0]
        if len(attack_indices) > 0:
            detected_attacks = np.sum(predictions[attack_indices] == 1)
            detection_rate = detected_attacks / len(attack_indices)
        else:
            detection_rate = 0.0
        
        # Calculate false positive rate
        normal_indices = np.where(y_true == 0)[0]
        if len(normal_indices) > 0:
            false_positives = np.sum(predictions[normal_indices] == 1)
            false_positive_rate = false_positives / len(normal_indices)
        else:
            false_positive_rate = 0.0
        
        results[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'predictions': predictions,
            'raw_predictions': raw_predictions
        }
    
    return results

def analyze_component_errors_during_attacks(residuals_data, output_dir=None):
    """
    Analyze component-level errors during attack points.
    
    Parameters
    ----------
    residuals_data : dict
        Dictionary containing residuals and ground truth
    output_dir : str, optional
        Directory to save analysis plots
        
    Returns
    -------
    dict
        Dictionary with component error analysis
    """
    print("\n=== Component Error Analysis During Attacks ===")
    
    # Extract data
    y_true = residuals_data['ground_truth']
    residuals_0 = residuals_data['residuals_0']  # Shape: (samples, components, features)
    residuals_1 = residuals_data['residuals_1']  # Shape: (samples, edges, features)
    residuals_2 = residuals_data['residuals_2']  # Shape: (samples, plcs, features)
    
    # Get attack indices
    attack_indices = np.where(y_true == 1)[0]
    normal_indices = np.where(y_true == 0)[0]
    
    print(f"Attack samples: {len(attack_indices)}")
    print(f"Normal samples: {len(normal_indices)}")
    
    # Analyze 0-cell (component) errors
    print(f"\n0-cell residuals shape: {residuals_0.shape}")
    print(f"1-cell residuals shape: {residuals_1.shape}")
    print(f"2-cell residuals shape: {residuals_2.shape}")
    
    # Calculate component-level statistics during attacks
    attack_residuals_0 = residuals_0[attack_indices]  # (attack_samples, components, features)
    normal_residuals_0 = residuals_0[normal_indices]  # (normal_samples, components, features)
    
    # Mean across feature dimensions for each component
    attack_comp_errors = np.mean(attack_residuals_0, axis=2)  # (attack_samples, components)
    normal_comp_errors = np.mean(normal_residuals_0, axis=2)  # (normal_samples, components)
    
    # Calculate statistics for each component
    comp_stats = {}
    num_components = attack_comp_errors.shape[1]
    
    for comp_idx in range(num_components):
        attack_errors = attack_comp_errors[:, comp_idx]
        normal_errors = normal_comp_errors[:, comp_idx]
        
        comp_stats[comp_idx] = {
            'attack_mean': np.mean(attack_errors),
            'attack_std': np.std(attack_errors),
            'normal_mean': np.mean(normal_errors),
            'normal_std': np.std(normal_errors),
            'error_ratio': np.mean(attack_errors) / (np.mean(normal_errors) + 1e-8),
            'attack_max': np.max(attack_errors),
            'normal_max': np.max(normal_errors)
        }
    
    # Find most affected components during attacks
    error_ratios = [comp_stats[i]['error_ratio'] for i in range(num_components)]
    most_affected_components = np.argsort(error_ratios)[::-1]  # Descending order
    
    print(f"\nTop 10 most affected components during attacks:")
    for i, comp_idx in enumerate(most_affected_components[:10]):
        stats = comp_stats[comp_idx]
        print(f"  Component {comp_idx}: Attack/Normal ratio = {stats['error_ratio']:.3f} "
              f"(Attack mean: {stats['attack_mean']:.4f}, Normal mean: {stats['normal_mean']:.4f})")
    
    # Analyze 1-cell and 2-cell errors
    if residuals_1.shape != residuals_0.shape:  # Not temporal mode
        attack_residuals_1 = residuals_1[attack_indices]
        normal_residuals_1 = residuals_1[normal_indices]
        attack_residuals_2 = residuals_2[attack_indices]
        normal_residuals_2 = residuals_2[normal_indices]
        
        # Calculate mean errors for edges and PLCs
        attack_edge_errors = np.mean(attack_residuals_1, axis=2)
        normal_edge_errors = np.mean(normal_residuals_1, axis=2)
        attack_plc_errors = np.mean(attack_residuals_2, axis=2)
        normal_plc_errors = np.mean(normal_residuals_2, axis=2)
        
        print(f"\n1-cell (edge) errors:")
        print(f"  Attack mean: {np.mean(attack_edge_errors):.4f}")
        print(f"  Normal mean: {np.mean(normal_edge_errors):.4f}")
        print(f"  Ratio: {np.mean(attack_edge_errors) / (np.mean(normal_edge_errors) + 1e-8):.3f}")
        
        print(f"\n2-cell (PLC) errors:")
        print(f"  Attack mean: {np.mean(attack_plc_errors):.4f}")
        print(f"  Normal mean: {np.mean(normal_plc_errors):.4f}")
        print(f"  Ratio: {np.mean(attack_plc_errors) / (np.mean(normal_plc_errors) + 1e-8):.3f}")
    else:
        print(f"\nTemporal mode detected - 1-cell and 2-cell residuals same as 0-cell")
    
    # Create visualization if output_dir provided
    if output_dir:
        import matplotlib.pyplot as plt
        
        # Plot component error distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Top affected components
        top_10_ratios = [comp_stats[i]['error_ratio'] for i in most_affected_components[:10]]
        axes[0, 0].bar(range(10), top_10_ratios)
        axes[0, 0].set_title('Top 10 Components: Attack/Normal Error Ratio')
        axes[0, 0].set_xlabel('Component Index')
        axes[0, 0].set_ylabel('Error Ratio')
        
        # Component error comparison
        attack_means = [comp_stats[i]['attack_mean'] for i in range(min(20, num_components))]
        normal_means = [comp_stats[i]['normal_mean'] for i in range(min(20, num_components))]
        x_pos = np.arange(len(attack_means))
        
        axes[0, 1].bar(x_pos - 0.2, attack_means, 0.4, label='Attack', alpha=0.7)
        axes[0, 1].bar(x_pos + 0.2, normal_means, 0.4, label='Normal', alpha=0.7)
        axes[0, 1].set_title('Component Error Comparison (First 20)')
        axes[0, 1].set_xlabel('Component Index')
        axes[0, 1].set_ylabel('Mean Error')
        axes[0, 1].legend()
        
        # Error distribution for top component
        if len(most_affected_components) > 0:
            top_comp = most_affected_components[0]
            axes[1, 0].hist(attack_comp_errors[:, top_comp], bins=50, alpha=0.7, label='Attack', density=True)
            axes[1, 0].hist(normal_comp_errors[:, top_comp], bins=50, alpha=0.7, label='Normal', density=True)
            axes[1, 0].set_title(f'Error Distribution: Component {top_comp} (Most Affected)')
            axes[1, 0].set_xlabel('Error Value')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].legend()
        
        # Cell type comparison
        if residuals_1.shape != residuals_0.shape:  # Not temporal mode
            cell_types = ['0-cells', '1-cells', '2-cells']
            attack_means = [
                np.mean(attack_comp_errors),
                np.mean(attack_edge_errors),
                np.mean(attack_plc_errors)
            ]
            normal_means = [
                np.mean(normal_comp_errors),
                np.mean(normal_edge_errors),
                np.mean(normal_plc_errors)
            ]
            
            x_pos = np.arange(len(cell_types))
            axes[1, 1].bar(x_pos - 0.2, attack_means, 0.4, label='Attack', alpha=0.7)
            axes[1, 1].bar(x_pos + 0.2, normal_means, 0.4, label='Normal', alpha=0.7)
            axes[1, 1].set_title('Cell Type Error Comparison')
            axes[1, 1].set_xlabel('Cell Type')
            axes[1, 1].set_ylabel('Mean Error')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(cell_types)
            axes[1, 1].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'component_error_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved component error analysis to {plot_path}")
        plt.show()
    
    return {
        'component_stats': comp_stats,
        'most_affected_components': most_affected_components,
        'attack_indices': attack_indices,
        'normal_indices': normal_indices,
        'attack_comp_errors': attack_comp_errors,
        'normal_comp_errors': normal_comp_errors
    }

def evaluate_cusum(residuals_data, S_values, G_values, temporal_consistency=1, norm_type='l2_squared'):
    """
    Evaluate CUSUM performance with different parameters.
    
    Parameters
    ----------
    residuals_data : dict
        Dictionary containing residuals and ground truth
    S_values : list
        List of S (scaling) parameter values to evaluate
    G_values : list
        List of G (growth) parameter values to evaluate
    temporal_consistency : int
        Number of consecutive anomalies required for detection
    norm_type : str
        Type of norm to use: 'l2_squared' (original), 'l2' (sqrt), 'l1' (absolute)
        
    Returns
    -------
    dict
        Dictionary with evaluation results for each parameter combination
    """
    y_true = residuals_data['ground_truth']
    residuals_0 = residuals_data['residuals_0']  # Full residual tensor (samples, features, dims)
    
    # Convert to 2D tensor for CUSUM (samples, features)
    if len(residuals_0.shape) == 3:
        # Take mean across feature dimensions
        residuals_2d = np.mean(residuals_0, axis=2)
    else:
        residuals_2d = residuals_0
    
    # Apply norm transformation based on norm_type
    if norm_type == 'l2':
        # Convert L2 squared to L2 (take square root)
        residuals_2d = np.sqrt(residuals_2d)
        print(f"Applied L2 norm (sqrt) transformation")
    elif norm_type == 'l1':
        # Convert L2 squared to L1 (approximate: sqrt of squared is absolute-like)
        # For residuals, this is more like taking sqrt to get magnitude
        residuals_2d = np.sqrt(residuals_2d)
        print(f"Applied L1-like norm transformation (sqrt)")
    else:  # 'l2_squared' (default)
        print(f"Using original L2 squared residuals")
    
    num_features = residuals_2d.shape[1]
    num_samples = residuals_2d.shape[0]
    
    print(f"Evaluating CUSUM with {len(S_values)} S values and {len(G_values)} G values")
    print(f"Residuals shape: {residuals_2d.shape}, Norm type: {norm_type}")
    
    results = {}
    
    # Split data into calibration (normal) and test sets
    normal_mask = (y_true == 0)
    test_mask = (y_true == 1)
    
    # Use first 70% of normal data for calibration
    normal_indices = np.where(normal_mask)[0]
    calib_size = int(0.7 * len(normal_indices))
    calib_indices = normal_indices[:calib_size]
    test_indices = np.concatenate([normal_indices[calib_size:], np.where(test_mask)[0]])
    
    print(f"Calibration samples: {len(calib_indices)}, Test samples: {len(test_indices)}")
    
    for S in S_values:
        for G in G_values:
            print(f"  Evaluating CUSUM S={S:.2f}, G={G:.2f}")
            
            # Initialize CUSUM
            cusum = CUSUM(num_features=num_features, S=S, G=G, device='cpu')
            
            # Calibrate on normal data
            calib_residuals = torch.tensor(residuals_2d[calib_indices], dtype=torch.float32)
            cusum.calibrate(calib_residuals)
            
            # Evaluate on test data
            predictions = []
            for i in test_indices:
                current_residual = torch.tensor(residuals_2d[i], dtype=torch.float32)
                anomalies = cusum.update(current_residual)
                # Consider it an anomaly if any feature is anomalous
                predictions.append(1 if torch.any(anomalies).item() else 0)
            
            predictions = np.array(predictions)
            test_labels = y_true[test_indices]
            
            # Apply temporal consistency if needed
            if temporal_consistency > 1:
                predictions = apply_temporal_consistency(predictions, temporal_consistency)
            
            # Calculate metrics
            precision = precision_score(test_labels, predictions, zero_division=0)
            recall = recall_score(test_labels, predictions, zero_division=0)
            f1 = f1_score(test_labels, predictions, zero_division=0)
            
            # Calculate detection rate
            attack_indices = np.where(test_labels == 1)[0]
            if len(attack_indices) > 0:
                detected_attacks = np.sum(predictions[attack_indices] == 1)
                detection_rate = detected_attacks / len(attack_indices)
            else:
                detection_rate = 0.0
            
            # Calculate false positive rate
            normal_test_indices = np.where(test_labels == 0)[0]
            if len(normal_test_indices) > 0:
                false_positives = np.sum(predictions[normal_test_indices] == 1)
                false_positive_rate = false_positives / len(normal_test_indices)
            else:
                false_positive_rate = 0.0
            
            results[(S, G)] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'detection_rate': detection_rate,
                'false_positive_rate': false_positive_rate,
                'predictions': predictions,
                'test_labels': test_labels
            }
    
    return results

def apply_temporal_consistency(predictions, consistency):
    """Apply temporal consistency filtering."""
    filtered = np.zeros_like(predictions)
    
    for i in range(len(predictions)):
        if i < consistency - 1:
            # Not enough history, use raw prediction
            filtered[i] = predictions[i]
        else:
            # Check if last 'consistency' predictions are all 1
            window = predictions[i-consistency+1:i+1]
            filtered[i] = 1 if np.all(window == 1) else 0
    
    return filtered

def compute_curves(residuals_data):
    """Compute PRC and ROC curves."""
    y_true = residuals_data['ground_truth']
    scores = residuals_data['error_per_sample_0']
    
    # Compute PRC
    precision, recall, prc_thresholds = precision_recall_curve(y_true, scores)
    average_precision = average_precision_score(y_true, scores)
    
    # Compute ROC
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    return {
        'prc': {'precision': precision, 'recall': recall, 'thresholds': prc_thresholds, 'ap': average_precision},
        'roc': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds, 'auc': roc_auc}
    }

def plot_curves(curves, save_path=None):
    """Plot PRC and ROC curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot PRC
    ax1.plot(curves['prc']['recall'], curves['prc']['precision'], 
             label=f'AP = {curves["prc"]["ap"]:.3f}', linewidth=2)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot ROC
    ax2.plot(curves['roc']['fpr'], curves['roc']['tpr'], 
             label=f'AUC = {curves["roc"]["auc"]:.3f}', linewidth=2)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved curves to {save_path}")
    
    plt.show()

def plot_threshold_analysis(results, save_path=None):
    """Plot threshold analysis."""
    thresholds = list(results.keys())
    metrics = ['precision', 'recall', 'f1', 'detection_rate', 'false_positive_rate']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results[t][metric] for t in thresholds]
        axes[i].plot(thresholds, values, 'o-', linewidth=2, markersize=4)
        axes[i].set_xlabel('Threshold')
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].set_title(f'{metric.replace("_", " ").title()} vs Threshold')
        axes[i].grid(True, alpha=0.3)
    
    # Remove the last subplot if we have odd number of metrics
    if len(metrics) < 6:
        axes[-1].remove()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved threshold analysis to {save_path}")
    
    plt.show()

def plot_cusum_results(cusum_results, save_path=None):
    """
    Plot CUSUM evaluation results.
    
    Parameters
    ----------
    cusum_results : dict
        Dictionary with CUSUM evaluation results
    save_path : str, optional
        Path to save the plot
    """
    # Extract S and G values and metrics
    S_values = sorted(list(set([S for S, G in cusum_results.keys()])))
    G_values = sorted(list(set([G for S, G in cusum_results.keys()])))
    
    # Create heatmaps for different metrics
    metrics = ['f1', 'precision', 'recall', 'detection_rate', 'false_positive_rate']
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        if i >= len(axes):
            break
            
        # Create heatmap data
        heatmap_data = np.zeros((len(G_values), len(S_values)))
        for j, G in enumerate(G_values):
            for k, S in enumerate(S_values):
                if (S, G) in cusum_results:
                    heatmap_data[j, k] = cusum_results[(S, G)][metric]
        
        # Plot heatmap
        im = axes[i].imshow(heatmap_data, cmap='viridis', aspect='auto')
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_xlabel('S (Scaling)')
        axes[i].set_ylabel('G (Growth)')
        
        # Set tick labels
        axes[i].set_xticks(range(len(S_values)))
        axes[i].set_xticklabels([f'{S:.2f}' for S in S_values], rotation=45)
        axes[i].set_yticks(range(len(G_values)))
        axes[i].set_yticklabels([f'{G:.2f}' for G in G_values])
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i])
    
    # Remove extra subplot if needed
    if len(metrics) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved CUSUM analysis to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection with different thresholds and CUSUM")
    parser.add_argument('--residuals', type=str, required=True, help='Path to residuals file')
    parser.add_argument('--config', type=str, help='Path to config file for experiment info')
    parser.add_argument('--temporal_consistency', type=int, default=1, help='Temporal consistency requirement')
    parser.add_argument('--output_dir', type=str, default='threshold_evaluation', help='Output directory')
    parser.add_argument('--log_to_wandb', action='store_true', help='Log results to wandb')
    parser.add_argument('--evaluate_cusum', action='store_true', help='Also evaluate CUSUM algorithm')
    parser.add_argument('--S_values', nargs='+', type=float, default=[1.0, 1.42, 2.0, 2.5, 3.0], 
                       help='CUSUM S (scaling) parameter values')
    parser.add_argument('--G_values', nargs='+', type=float, default=[3.0, 5.98, 8.0, 10.0], 
                       help='CUSUM G (growth) parameter values')
    parser.add_argument('--norm_type', type=str, default='l2_squared', choices=['l2_squared', 'l2', 'l1'],
                       help='Type of norm to use for CUSUM evaluation (l2_squared, l2, l1)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load residuals
    residuals_data = load_residuals(args.residuals)
    
    # Generate threshold range
    scores = residuals_data['error_per_sample_0']
    min_score, max_score = np.min(scores), np.max(scores)
    
    # Create threshold range (logarithmic spacing for better coverage)
    thresholds = np.logspace(np.log10(min_score), np.log10(max_score), 100)
    
    print(f"Evaluating {len(thresholds)} thresholds from {min_score:.6f} to {max_score:.6f}")
    
    # Evaluate thresholds
    results = evaluate_thresholds(residuals_data, thresholds, args.temporal_consistency)
    
    # Compute curves
    curves = compute_curves(residuals_data)
    
    # Plot curves
    curves_path = os.path.join(args.output_dir, 'curves.png')
    plot_curves(curves, curves_path)
    
    # Plot threshold analysis
    analysis_path = os.path.join(args.output_dir, 'threshold_analysis.png')
    plot_threshold_analysis(results, analysis_path)
    
    # Find best threshold by F1 score
    best_threshold = max(results.keys(), key=lambda t: results[t]['f1'])
    best_results = results[best_threshold]
    
    print(f"\nBest threshold by F1 score: {best_threshold:.6f}")
    print(f"Best F1: {best_results['f1']:.4f}")
    print(f"Precision: {best_results['precision']:.4f}")
    print(f"Recall: {best_results['recall']:.4f}")
    print(f"Detection Rate: {best_results['detection_rate']:.4f}")
    print(f"False Positive Rate: {best_results['false_positive_rate']:.4f}")
    
    # Analyze component errors during attacks
    analyze_component_errors_during_attacks(residuals_data, args.output_dir)

    # Evaluate CUSUM if requested
    cusum_results = None
    if args.evaluate_cusum:
        print(f"\nEvaluating CUSUM with S={args.S_values}, G={args.G_values}, Norm={args.norm_type}")
        cusum_results = evaluate_cusum(residuals_data, args.S_values, args.G_values, args.temporal_consistency, args.norm_type)
        
        # Find best CUSUM parameters by F1 score
        best_cusum_params = max(cusum_results.keys(), key=lambda params: cusum_results[params]['f1'])
        best_cusum_results = cusum_results[best_cusum_params]
        
        print(f"\nBest CUSUM parameters: S={best_cusum_params[0]:.2f}, G={best_cusum_params[1]:.2f}")
        print(f"Best CUSUM F1: {best_cusum_results['f1']:.4f}")
        print(f"Best CUSUM Precision: {best_cusum_results['precision']:.4f}")
        print(f"Best CUSUM Recall: {best_cusum_results['recall']:.4f}")
        print(f"Best CUSUM Detection Rate: {best_cusum_results['detection_rate']:.4f}")
        print(f"Best CUSUM False Positive Rate: {best_cusum_results['false_positive_rate']:.4f}")
        
        # Plot CUSUM results
        cusum_analysis_path = os.path.join(args.output_dir, 'cusum_analysis.png')
        plot_cusum_results(cusum_results, cusum_analysis_path)
    
    # Save results
    evaluation_results = {
        'curves': curves,
        'threshold_results': results,
        'best_threshold': best_threshold,
        'best_results': best_results,
        'dataset_info': residuals_data['dataset_info'],
        'temporal_consistency': args.temporal_consistency,
        'cusum_results': cusum_results
    }
    
    results_path = os.path.join(args.output_dir, 'evaluation_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(evaluation_results, f)
    
    print(f"Saved evaluation results to {results_path}")
    
    # Log to wandb if requested
    if args.log_to_wandb:
        if args.config:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            
            wandb.init(
                project=config.get('project_name', 'threshold-evaluation'),
                name=f"threshold_eval_{config.get('experiment_name', 'unknown')}",
                config=config
            )
            
            # Log metrics
            wandb.log({
                'best_threshold': best_threshold,
                'best_f1': best_results['f1'],
                'best_precision': best_results['precision'],
                'best_recall': best_results['recall'],
                'best_detection_rate': best_results['detection_rate'],
                'best_false_positive_rate': best_results['false_positive_rate'],
                'prc_ap': curves['prc']['ap'],
                'roc_auc': curves['roc']['auc'],
                'temporal_consistency': args.temporal_consistency
            })
            
            # Log CUSUM metrics if evaluated
            if cusum_results:
                best_cusum_params = max(cusum_results.keys(), key=lambda params: cusum_results[params]['f1'])
                best_cusum_results = cusum_results[best_cusum_params]
                
                wandb.log({
                    'best_cusum_S': best_cusum_params[0],
                    'best_cusum_G': best_cusum_params[1],
                    'best_cusum_f1': best_cusum_results['f1'],
                    'best_cusum_precision': best_cusum_results['precision'],
                    'best_cusum_recall': best_cusum_results['recall'],
                    'best_cusum_detection_rate': best_cusum_results['detection_rate'],
                    'best_cusum_false_positive_rate': best_cusum_results['false_positive_rate']
                })
            
            # Log plots
            wandb.log({
                'curves': wandb.Image(curves_path),
                'threshold_analysis': wandb.Image(analysis_path)
            })
            
            if cusum_results:
                wandb.log({
                    'cusum_analysis': wandb.Image(cusum_analysis_path)
                })
            
            wandb.finish()

if __name__ == '__main__':
    main() 