"""
Debug analysis utilities for investigating MinMax normalization collapse issues.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd


def analyze_feature_distributions(data, columns, normalization_method="minmax_proper"):
    """
    Analyze feature distributions before and after normalization.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Raw data
    columns : list
        Feature columns to analyze
    normalization_method : str
        Normalization method to test
        
    Returns
    -------
    dict
        Analysis results
    """
    print(f"=== Feature Distribution Analysis ({normalization_method}) ===")
    
    results = {
        'raw_stats': {},
        'normalized_stats': {},
        'constant_features': [],
        'high_variance_features': [],
        'low_variance_features': []
    }
    
    # Analyze raw features
    for col in columns:
        values = data[col].values
        
        # Raw statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        range_val = max_val - min_val
        
        results['raw_stats'][col] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'range': range_val,
            'cv': std_val / (abs(mean_val) + 1e-8)  # Coefficient of variation
        }
        
        # Classify features
        if range_val == 0:
            results['constant_features'].append(col)
        elif std_val / (abs(mean_val) + 1e-8) > 0.5:
            results['high_variance_features'].append(col)
        elif std_val / (abs(mean_val) + 1e-8) < 0.1:
            results['low_variance_features'].append(col)
    
    # Test normalization
    if normalization_method == "minmax_proper":
        scaler = MinMaxScaler()
        normalized_values = scaler.fit_transform(data[columns])
    elif normalization_method == "z_normalization_proper":
        scaler = StandardScaler()
        normalized_values = scaler.fit_transform(data[columns])
    else:
        print(f"Unknown normalization method: {normalization_method}")
        return results
    
    # Analyze normalized features
    for i, col in enumerate(columns):
        norm_values = normalized_values[:, i]
        
        results['normalized_stats'][col] = {
            'mean': np.mean(norm_values),
            'std': np.std(norm_values),
            'min': np.min(norm_values),
            'max': np.max(norm_values),
            'range': np.max(norm_values) - np.min(norm_values)
        }
    
    # Print summary
    print(f"Total features: {len(columns)}")
    print(f"Constant features: {len(results['constant_features'])}")
    print(f"High variance features: {len(results['high_variance_features'])}")
    print(f"Low variance features: {len(results['low_variance_features'])}")
    
    if results['constant_features']:
        print(f"Constant features: {results['constant_features'][:5]}...")
    
    return results


def compare_normalization_methods(data, columns, sample_size=1000):
    """
    Compare different normalization methods on the same data.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Raw data
    columns : list
        Feature columns to analyze
    sample_size : int
        Number of samples to use for analysis
        
    Returns
    -------
    dict
        Comparison results
    """
    print(f"=== Normalization Method Comparison ===")
    
    # Sample data for analysis
    if len(data) > sample_size:
        sample_data = data.sample(n=sample_size, random_state=42)
    else:
        sample_data = data
    
    methods = {
        'minmax_proper': MinMaxScaler(),
        'z_normalization_proper': StandardScaler(),
        'robust': None  # Will implement robust scaling
    }
    
    results = {}
    
    for method_name, scaler in methods.items():
        print(f"\n--- {method_name.upper()} ---")
        
        if scaler is not None:
            normalized_values = scaler.fit_transform(sample_data[columns])
        else:
            # Robust scaling (median and IQR)
            normalized_values = np.zeros_like(sample_data[columns].values)
            for i, col in enumerate(columns):
                values = sample_data[col].values
                median_val = np.median(values)
                q75, q25 = np.percentile(values, [75, 25])
                iqr = q75 - q25
                if iqr == 0:
                    normalized_values[:, i] = 0
                else:
                    normalized_values[:, i] = (values - median_val) / iqr
        
        # Calculate statistics
        results[method_name] = {
            'mean': np.mean(normalized_values, axis=0),
            'std': np.std(normalized_values, axis=0),
            'min': np.min(normalized_values, axis=0),
            'max': np.max(normalized_values, axis=0),
            'range': np.max(normalized_values, axis=0) - np.min(normalized_values, axis=0),
            'zeros': np.sum(normalized_values == 0, axis=0),
            'unique_values': [len(np.unique(normalized_values[:, i])) for i in range(len(columns))]
        }
        
        print(f"  Mean range: [{np.mean(results[method_name]['mean']):.4f}, {np.mean(results[method_name]['mean']):.4f}]")
        print(f"  Std range: [{np.mean(results[method_name]['std']):.4f}, {np.mean(results[method_name]['std']):.4f}]")
        print(f"  Value range: [{np.mean(results[method_name]['min']):.4f}, {np.mean(results[method_name]['max']):.4f}]")
        print(f"  Average unique values: {np.mean(results[method_name]['unique_values']):.1f}")
        print(f"  Zero values: {np.sum(results[method_name]['zeros'])} total")
    
    return results


def analyze_reconstruction_errors(model, dataloader, device, threshold_percentile=99):
    """
    Analyze reconstruction errors to understand threshold calibration issues.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained model
    dataloader : torch.utils.data.DataLoader
        Data loader for evaluation
    device : torch.device
        Device to run analysis on
    threshold_percentile : float
        Percentile for threshold calculation
        
    Returns
    -------
    dict
        Error analysis results
    """
    print(f"=== Reconstruction Error Analysis ===")
    
    model.eval()
    all_errors_0 = []
    all_errors_1 = []
    all_errors_2 = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 1000 == 0:
                print(f"Processing batch {batch_idx}")
            
            # Extract features
            if isinstance(batch, (list, tuple)):
                x_0, x_1, x_2, a0, a1, coa2, b1, b2, _ = batch
            else:
                x_0 = batch['x_0']
                x_1 = batch['x_1'] 
                x_2 = batch['x_2']
                a0 = batch['a0']
                a1 = batch['a1']
                coa2 = batch['coa2']
                b1 = batch['b1']
                b2 = batch['b2']
            
            # Move to device
            x_0, x_1, x_2 = x_0.to(device), x_1.to(device), x_2.to(device)
            a0, a1, coa2 = a0.to(device), a1.to(device), coa2.to(device)
            b1, b2 = b1.to(device), b2.to(device)
            
            # Forward pass
            recon_x0, recon_x1, recon_x2 = model(x_0, x_1, x_2, a0, a1, coa2, b1, b2)
            
            # Compute errors
            errors_0 = torch.mean((recon_x0 - x_0) ** 2, dim=2)  # [batch, n_0cells]
            errors_1 = torch.mean((recon_x1 - x_1) ** 2, dim=2)  # [batch, n_1cells]
            errors_2 = torch.mean((recon_x2 - x_2) ** 2, dim=2)  # [batch, n_2cells]
            
            all_errors_0.append(errors_0.cpu())
            all_errors_1.append(errors_1.cpu())
            all_errors_2.append(errors_2.cpu())
    
    # Concatenate all errors
    all_errors_0 = torch.cat(all_errors_0, dim=0)  # [total_samples, n_0cells]
    all_errors_1 = torch.cat(all_errors_1, dim=0)  # [total_samples, n_1cells]
    all_errors_2 = torch.cat(all_errors_2, dim=0)  # [total_samples, n_2cells]
    
    # Calculate statistics
    results = {
        'errors_0': all_errors_0.numpy(),
        'errors_1': all_errors_1.numpy(),
        'errors_2': all_errors_2.numpy(),
        'stats_0': {
            'mean': np.mean(all_errors_0.numpy(), axis=0),
            'std': np.std(all_errors_0.numpy(), axis=0),
            'min': np.min(all_errors_0.numpy(), axis=0),
            'max': np.max(all_errors_0.numpy(), axis=0),
            'percentile_99': np.percentile(all_errors_0.numpy(), threshold_percentile, axis=0)
        },
        'stats_1': {
            'mean': np.mean(all_errors_1.numpy(), axis=0),
            'std': np.std(all_errors_1.numpy(), axis=0),
            'min': np.min(all_errors_1.numpy(), axis=0),
            'max': np.max(all_errors_1.numpy(), axis=0),
            'percentile_99': np.percentile(all_errors_1.numpy(), threshold_percentile, axis=0)
        },
        'stats_2': {
            'mean': np.mean(all_errors_2.numpy(), axis=0),
            'std': np.std(all_errors_2.numpy(), axis=0),
            'min': np.min(all_errors_2.numpy(), axis=0),
            'max': np.max(all_errors_2.numpy(), axis=0),
            'percentile_99': np.percentile(all_errors_2.numpy(), threshold_percentile, axis=0)
        }
    }
    
    # Print summary
    print(f"Error Analysis Summary:")
    print(f"  0-cells: mean={np.mean(results['stats_0']['mean']):.6f}, std={np.mean(results['stats_0']['std']):.6f}")
    print(f"  1-cells: mean={np.mean(results['stats_1']['mean']):.6f}, std={np.mean(results['stats_1']['std']):.6f}")
    print(f"  2-cells: mean={np.mean(results['stats_2']['mean']):.6f}, std={np.mean(results['stats_2']['std']):.6f}")
    
    return results


def plot_error_distributions(error_results, save_path=None):
    """
    Plot error distributions for analysis.
    
    Parameters
    ----------
    error_results : dict
        Results from analyze_reconstruction_errors
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot error distributions for each cell type
    for i, (cell_type, errors) in enumerate([('0-cells', error_results['errors_0']), 
                                           ('1-cells', error_results['errors_1']), 
                                           ('2-cells', error_results['errors_2'])]):
        
        # Histogram of mean errors per sample
        mean_errors = np.mean(errors, axis=1)
        axes[0, i].hist(mean_errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0, i].set_title(f'{cell_type} - Mean Error Distribution')
        axes[0, i].set_xlabel('Mean Error')
        axes[0, i].set_ylabel('Frequency')
        
        # Box plot of errors per component
        axes[1, i].boxplot(errors.T, showfliers=False)
        axes[1, i].set_title(f'{cell_type} - Error Distribution per Component')
        axes[1, i].set_xlabel('Component Index')
        axes[1, i].set_ylabel('Error')
        axes[1, i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution plot saved to {save_path}")
    
    plt.show()


def suggest_threshold_strategy(error_results, data_info):
    """
    Suggest threshold strategy based on error analysis.
    
    Parameters
    ----------
    error_results : dict
        Results from analyze_reconstruction_errors
    data_info : dict
        Information about the dataset
        
    Returns
    -------
    dict
        Suggested threshold strategy
    """
    print(f"=== Threshold Strategy Suggestions ===")
    
    suggestions = {
        'global_threshold': {},
        'per_component_threshold': {},
        'adaptive_threshold': {},
        'recommendations': []
    }
    
    # Analyze error distributions
    for cell_type in ['0', '1', '2']:
        errors = error_results[f'errors_{cell_type}']
        stats = error_results[f'stats_{cell_type}']
        
        # Global threshold analysis
        global_mean = np.mean(stats['mean'])
        global_std = np.mean(stats['std'])
        global_percentile_99 = np.mean(stats['percentile_99'])
        
        suggestions['global_threshold'][cell_type] = {
            'mean_plus_2std': global_mean + 2 * global_std,
            'percentile_99': global_percentile_99,
            'percentile_95': np.percentile(errors, 95),
            'percentile_90': np.percentile(errors, 90)
        }
        
        # Per-component threshold analysis
        component_thresholds = stats['percentile_99']
        suggestions['per_component_threshold'][cell_type] = {
            'thresholds': component_thresholds,
            'mean_threshold': np.mean(component_thresholds),
            'std_threshold': np.std(component_thresholds),
            'min_threshold': np.min(component_thresholds),
            'max_threshold': np.max(component_thresholds)
        }
        
        # Check for threshold issues
        if global_percentile_99 < 0.01:
            suggestions['recommendations'].append(
                f"WARNING: {cell_type}-cells have very low 99th percentile threshold ({global_percentile_99:.6f}). "
                f"This may cause over-detection. Consider per-component thresholds."
            )
        
        if np.std(component_thresholds) > np.mean(component_thresholds):
            suggestions['recommendations'].append(
                f"SUGGESTION: {cell_type}-cells have high threshold variance. "
                f"Per-component thresholds may be more effective."
            )
    
    # Print suggestions
    print("Threshold Analysis:")
    for cell_type in ['0', '1', '2']:
        print(f"\n{cell_type}-cells:")
        print(f"  Global 99th percentile: {suggestions['global_threshold'][cell_type]['percentile_99']:.6f}")
        print(f"  Per-component mean: {suggestions['per_component_threshold'][cell_type]['mean_threshold']:.6f}")
        print(f"  Per-component std: {suggestions['per_component_threshold'][cell_type]['std_threshold']:.6f}")
    
    print(f"\nRecommendations:")
    for rec in suggestions['recommendations']:
        print(f"  {rec}")
    
    return suggestions 


def verify_normalization_consistency(train_dataset, validation_dataset, test_dataset):
    """
    Verify that normalization parameters are consistent across datasets.
    
    Parameters
    ----------
    train_dataset : SWaTDataset
        Training dataset
    validation_dataset : SWaTDataset
        Validation dataset
    test_dataset : SWaTDataset
        Test dataset
        
    Returns
    -------
    bool
        True if parameters are consistent, False otherwise
    """
    print("=== Normalization Parameter Consistency Check ===")
    
    datasets = {
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset
    }
    
    # Check which normalization method is being used
    normalization_method = train_dataset.normalization_method
    print(f"Normalization method: {normalization_method}")
    
    if normalization_method in ['minmax_proper']:
        param_names = ['train_min_vals', 'train_max_vals']
        param_display_names = ['Min Values', 'Max Values']
    elif normalization_method in ['z_normalization_proper']:
        param_names = ['train_mean_vals', 'train_std_vals']
        param_display_names = ['Mean Values', 'Std Values']
    else:
        print(f"Standard normalization method '{normalization_method}' - no parameters to check")
        return True
    
    # Check parameter availability
    for dataset_name, dataset in datasets.items():
        if dataset is None:
            continue
            
        print(f"\n{dataset_name.upper()} Dataset:")
        for param_name, display_name in zip(param_names, param_display_names):
            if hasattr(dataset, param_name):
                param_value = getattr(dataset, param_name)
                print(f"  {display_name}: {len(param_value)} parameters")
                if len(param_value) > 0:
                    print(f"    Range: [{min(param_value):.6f}, {max(param_value):.6f}]")
                    print(f"    Mean: {np.mean(param_value):.6f}")
            else:
                print(f"  {display_name}: NOT FOUND")
    
    # Check consistency between datasets
    print(f"\nConsistency Check:")
    reference_dataset = train_dataset
    
    for dataset_name, dataset in datasets.items():
        if dataset is None or dataset_name == 'train':
            continue
            
        is_consistent = True
        for param_name in param_names:
            if hasattr(reference_dataset, param_name) and hasattr(dataset, param_name):
                ref_params = getattr(reference_dataset, param_name)
                test_params = getattr(dataset, param_name)
                
                if len(ref_params) != len(test_params):
                    print(f"  ❌ {dataset_name}: Parameter count mismatch for {param_name}")
                    is_consistent = False
                elif not np.allclose(ref_params, test_params, rtol=1e-6):
                    print(f"  ❌ {dataset_name}: Parameter values mismatch for {param_name}")
                    max_diff = np.max(np.abs(np.array(ref_params) - np.array(test_params)))
                    print(f"    Max difference: {max_diff:.6f}")
                    is_consistent = False
                else:
                    print(f"  ✅ {dataset_name}: Parameters consistent for {param_name}")
            else:
                print(f"  ❌ {dataset_name}: Missing parameters for {param_name}")
                is_consistent = False
        
        if is_consistent:
            print(f"  ✅ {dataset_name.upper()} dataset: ALL PARAMETERS CONSISTENT")
        else:
            print(f"  ❌ {dataset_name.upper()} dataset: PARAMETER INCONSISTENCIES DETECTED")
    
    return is_consistent


def analyze_feature_statistics_by_dataset(train_dataset, validation_dataset, test_dataset, sample_size=100):
    """
    Analyze feature statistics across datasets to detect normalization issues.
    
    Parameters
    ----------
    train_dataset : SWaTDataset
        Training dataset
    validation_dataset : SWaTDataset
        Validation dataset  
    test_dataset : SWaTDataset
        Test dataset
    sample_size : int
        Number of samples to analyze from each dataset
        
    Returns
    -------
    dict
        Analysis results
    """
    print("=== Feature Statistics Analysis Across Datasets ===")
    
    datasets = {
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset
    }
    
    results = {}
    
    for dataset_name, dataset in datasets.items():
        if dataset is None:
            continue
            
        print(f"\n{dataset_name.upper()} Dataset:")
        
        # Sample data for analysis
        if len(dataset) > sample_size:
            indices = np.random.choice(len(dataset), sample_size, replace=False)
        else:
            indices = np.arange(len(dataset))
        
        # Extract features
        x0_samples = []
        x1_samples = []
        x2_samples = []
        
        for idx in indices:
            sample = dataset[idx]
            if isinstance(sample, (list, tuple)):
                x0_samples.append(sample[0])
                x1_samples.append(sample[1])
                x2_samples.append(sample[2])
            else:
                x0_samples.append(sample['x_0'])
                x1_samples.append(sample['x_1'])
                x2_samples.append(sample['x_2'])
        
        # Convert to numpy arrays
        x0_array = torch.stack(x0_samples).numpy()
        x1_array = torch.stack(x1_samples).numpy()
        x2_array = torch.stack(x2_samples).numpy()
        
        # Calculate statistics
        results[dataset_name] = {
            'x0': {
                'mean': np.mean(x0_array, axis=(0, 1)),  # Average across samples and components
                'std': np.std(x0_array, axis=(0, 1)),
                'min': np.min(x0_array, axis=(0, 1)),
                'max': np.max(x0_array, axis=(0, 1)),
                'range': np.max(x0_array, axis=(0, 1)) - np.min(x0_array, axis=(0, 1))
            },
            'x1': {
                'mean': np.mean(x1_array, axis=(0, 1)),
                'std': np.std(x1_array, axis=(0, 1)),
                'min': np.min(x1_array, axis=(0, 1)),
                'max': np.max(x1_array, axis=(0, 1)),
                'range': np.max(x1_array, axis=(0, 1)) - np.min(x1_array, axis=(0, 1))
            },
            'x2': {
                'mean': np.mean(x2_array, axis=(0, 1)),
                'std': np.std(x2_array, axis=(0, 1)),
                'min': np.min(x2_array, axis=(0, 1)),
                'max': np.max(x2_array, axis=(0, 1)),
                'range': np.max(x2_array, axis=(0, 1)) - np.min(x2_array, axis=(0, 1))
            }
        }
        
        # Print summary
        for cell_type, stats in results[dataset_name].items():
            print(f"  {cell_type}: mean={float(np.mean(stats['mean'])):.4f}, std={float(np.mean(stats['std'])):.4f}, range=[{float(np.min(stats['min'])):.4f}, {float(np.max(stats['max'])):.4f}]")
    
    # Check for significant differences
    print(f"\nConsistency Analysis:")
    reference_stats = results['train']
    
    for dataset_name, dataset_stats in results.items():
        if dataset_name == 'train':
            continue
            
        print(f"\n{dataset_name.upper()} vs TRAIN:")
        for cell_type in ['x0', 'x1', 'x2']:
            ref_stats = reference_stats[cell_type]
            test_stats = dataset_stats[cell_type]
            
            mean_diff = abs(np.mean(ref_stats['mean']) - np.mean(test_stats['mean']))
            std_diff = abs(np.mean(ref_stats['std']) - np.mean(test_stats['std']))
            
            if mean_diff > 0.1 or std_diff > 0.1:
                print(f"  ⚠️  {cell_type}: Significant difference detected")
                print(f"    Mean diff: {mean_diff:.4f}")
                print(f"    Std diff: {std_diff:.4f}")
            else:
                print(f"  ✅ {cell_type}: Statistics consistent")
    
    return results 


def investigate_extreme_values(dataset, sample_size=100):
    """
    Investigate extreme values in feature tensors to identify problematic components.
    
    Parameters
    ----------
    dataset : SWaTDataset
        Dataset to investigate
    sample_size : int
        Number of samples to analyze
        
    Returns
    -------
    dict
        Analysis results
    """
    print(f"=== Investigating Extreme Values in Dataset ===")
    
    # Sample data for analysis
    if len(dataset) > sample_size:
        indices = np.random.choice(len(dataset), sample_size, replace=False)
    else:
        indices = np.arange(len(dataset))
    
    # Extract features
    x0_samples = []
    for idx in indices:
        sample = dataset[idx]
        if isinstance(sample, (list, tuple)):
            x0_samples.append(sample[0])
        else:
            x0_samples.append(sample['x_0'])
    
    # Convert to numpy array
    x0_array = torch.stack(x0_samples).numpy()  # [samples, components, features]
    
    print(f"Feature tensor shape: {x0_array.shape}")
    print(f"Overall stats: min={x0_array.min():.6f}, max={x0_array.max():.6f}, mean={x0_array.mean():.6f}")
    
    # Check for extreme values in each component
    extreme_components = []
    for comp_idx in range(x0_array.shape[1]):
        comp_data = x0_array[:, comp_idx, :]  # [samples, features]
        
        comp_min = comp_data.min()
        comp_max = comp_data.max()
        comp_mean = comp_data.mean()
        comp_std = comp_data.std()
        
        # Check for extreme values
        if abs(comp_min) > 1000 or abs(comp_max) > 1000 or abs(comp_mean) > 100:
            extreme_components.append({
                'index': comp_idx,
                'name': dataset.columns[comp_idx] if hasattr(dataset, 'columns') else f'Component_{comp_idx}',
                'min': comp_min,
                'max': comp_max,
                'mean': comp_mean,
                'std': comp_std,
                'feature_0_min': comp_data[:, 0].min(),
                'feature_0_max': comp_data[:, 0].max(),
                'feature_0_mean': comp_data[:, 0].mean(),
                'feature_1_min': comp_data[:, 1].min() if comp_data.shape[1] > 1 else 0,
                'feature_1_max': comp_data[:, 1].max() if comp_data.shape[1] > 1 else 0,
                'feature_1_mean': comp_data[:, 1].mean() if comp_data.shape[1] > 1 else 0
            })
    
    # Print extreme components
    if extreme_components:
        print(f"\n🚨 Found {len(extreme_components)} components with extreme values:")
        for comp in extreme_components[:10]:  # Show first 10
            print(f"  {comp['name']} (idx={comp['index']}):")
            print(f"    Overall: min={comp['min']:.6f}, max={comp['max']:.6f}, mean={comp['mean']:.6f}, std={comp['std']:.6f}")
            print(f"    Feature 0: min={comp['feature_0_min']:.6f}, max={comp['feature_0_max']:.6f}, mean={comp['feature_0_mean']:.6f}")
            if comp_data.shape[1] > 1:
                print(f"    Feature 1: min={comp['feature_1_min']:.6f}, max={comp['feature_1_max']:.6f}, mean={comp['feature_1_mean']:.6f}")
    else:
        print("✅ No extreme values detected")
    
    # Check for NaN or infinite values
    nan_count = np.isnan(x0_array).sum()
    inf_count = np.isinf(x0_array).sum()
    
    if nan_count > 0:
        print(f"🚨 Found {nan_count} NaN values!")
    if inf_count > 0:
        print(f"🚨 Found {inf_count} infinite values!")
    
    return {
        'extreme_components': extreme_components,
        'nan_count': nan_count,
        'inf_count': inf_count,
        'overall_stats': {
            'min': x0_array.min(),
            'max': x0_array.max(),
            'mean': x0_array.mean(),
            'std': x0_array.std()
        }
    }


def debug_normalization_step_by_step(dataset, component_name, sample_size=10):
    """
    Debug normalization step by step for a specific component.
    
    Parameters
    ----------
    dataset : SWaTDataset
        Dataset to debug
    component_name : str
        Name of the component to debug
    sample_size : int
        Number of samples to analyze
    """
    print(f"=== Debugging Normalization for {component_name} ===")
    
    # Find component index
    if not hasattr(dataset, 'columns'):
        print("❌ Dataset doesn't have columns attribute")
        return
    
    try:
        comp_idx = dataset.columns.index(component_name)
    except ValueError:
        print(f"❌ Component {component_name} not found in dataset")
        return
    
    # Get raw values
    raw_values = dataset.data[component_name].values[:sample_size]
    print(f"Raw values (first {sample_size}): {raw_values}")
    print(f"Raw stats: min={raw_values.min():.6f}, max={raw_values.max():.6f}, mean={raw_values.mean():.6f}, std={raw_values.std():.6f}")
    
    # Check normalization parameters
    if hasattr(dataset, 'train_mean_vals') and hasattr(dataset, 'train_std_vals'):
        if comp_idx < len(dataset.train_mean_vals):
            train_mean = dataset.train_mean_vals[comp_idx]
            train_std = dataset.train_std_vals[comp_idx]
            print(f"Training params: mean={train_mean:.6f}, std={train_std:.6f}")
            
            # Manual normalization
            manual_normalized = (raw_values - train_mean) / train_std
            print(f"Manual normalized: min={manual_normalized.min():.6f}, max={manual_normalized.max():.6f}, mean={manual_normalized.mean():.6f}")
        else:
            print(f"❌ Component index {comp_idx} out of range for training parameters")
    else:
        print("❌ No training parameters found")
    
    # Get actual normalized values from dataset
    sample_indices = np.arange(min(sample_size, len(dataset)))
    actual_normalized = []
    
    for idx in sample_indices:
        sample = dataset[idx]
        if isinstance(sample, (list, tuple)):
            actual_normalized.append(sample[0][comp_idx].numpy())
        else:
            actual_normalized.append(sample['x_0'][comp_idx].numpy())
    
    actual_normalized = np.array(actual_normalized)
    print(f"Actual normalized (first {sample_size}):")
    print(f"  Feature 0: min={actual_normalized[:, 0].min():.6f}, max={actual_normalized[:, 0].max():.6f}, mean={actual_normalized[:, 0].mean():.6f}")
    if actual_normalized.shape[1] > 1:
        print(f"  Feature 1: min={actual_normalized[:, 1].min():.6f}, max={actual_normalized[:, 1].max():.6f}, mean={actual_normalized[:, 1].mean():.6f}") 