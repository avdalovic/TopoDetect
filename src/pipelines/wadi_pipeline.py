import os
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch

# Imports for WADI
from src.utils.topology.wadi_topology import WADIComplex
from src.utils.attack_utils import get_attack_indices, get_attack_sds
from src.datasets.wadi_dataset import WADIDataset
from src.models.ccann import AnomalyCCANN
from src.trainers.anomaly_trainer import AnomalyTrainer


def load_wadi_data(train_path, test_path, sample_rate=1.0, save_test_path=None, validation_split_ratio=0.2, 
                  attack_focused_sampling=False, pre_attack_points=10, attack_start_points=5):
    """
    Load WADI dataset from CSV files with optional sampling and validation split.

    Parameters
    ----------
    train_path : str
        Path to training data CSV file
    test_path : str
        Path to test data CSV file
    sample_rate : float, default=1.0
        Fraction of data to sample (0.0 to 1.0)
    save_test_path : str, optional
        Path to save processed test data
    validation_split_ratio : float, default=0.2
        Fraction of training data to use for validation
    attack_focused_sampling : bool, default=False
        If True, sample points around attack transitions to preserve temporal coherence
    pre_attack_points : int, default=10
        Number of points to include before each attack when using attack_focused_sampling
    attack_start_points : int, default=5
        Number of points to include from the beginning of each attack

    Returns
    -------
    tuple of pandas.DataFrame
        (train_data, validation_data, test_data)
    """
    print(f"Loading WADI data from {train_path} and {test_path}...")
    print(f"Using sample rate: {sample_rate}")
    print(f"Using validation split ratio: {validation_split_ratio}")

    # Load data
    initial_train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Apply sampling to training data (always use linspace approach)
    if sample_rate < 1.0:
        # Sample training data using the original method
        train_sample_size = int(len(initial_train_data) * sample_rate)
        train_indices = np.linspace(0, len(initial_train_data)-1, train_sample_size, dtype=int)
        initial_train_data = initial_train_data.iloc[train_indices].reset_index(drop=True)
        
        # Apply sampling to test data based on chosen method
        if attack_focused_sampling:
            # Attack-focused sampling for test data only
            print(f"Using attack-focused sampling for test data with {pre_attack_points} pre-attack points and {attack_start_points} attack-start points")
            # Get attack indices from attack_utils
            attack_indices, _ = get_attack_indices("WADI")
            
            # Calculate how many total points we want based on sample_rate
            test_sample_size = int(len(test_data) * sample_rate)
            
            # For test data, focus sampling around attack transitions
            selected_indices = []
            
            # Add points around each attack
            for attack_range in attack_indices:
                attack_start = attack_range[0]  # First point of the attack
                
                # Add points before attack starts (if possible)
                pre_attack_start = max(0, attack_start - pre_attack_points)
                selected_indices.extend(range(pre_attack_start, attack_start))
                
                # Add points from beginning of attack
                attack_end_sample = min(attack_start + attack_start_points, attack_range[-1] + 1)
                selected_indices.extend(range(attack_start, attack_end_sample))
            
            # If we haven't selected enough points, add more evenly from the dataset
            if len(selected_indices) < test_sample_size:
                remaining_points = test_sample_size - len(selected_indices)
                # Create a mask of indices we haven't selected yet
                all_indices = set(range(len(test_data)))
                available_indices = list(all_indices - set(selected_indices))
                
                # Sample evenly from remaining points
                if len(available_indices) > remaining_points:
                    additional_indices = np.linspace(0, len(available_indices)-1, remaining_points, dtype=int)
                    selected_indices.extend([available_indices[i] for i in additional_indices])
                else:
                    # If we need more points than available, just take what's left
                    selected_indices.extend(available_indices)
            
            # If we have too many points, subsample
            if len(selected_indices) > test_sample_size:
                # Prioritize keeping attack transition points by removing others
                attack_indices_flat = set()
                for attack in attack_indices:
                    attack_start = attack[0]
                    pre_attack_start = max(0, attack_start - pre_attack_points)
                    attack_end_sample = min(attack_start + attack_start_points, attack[-1] + 1)
                    for i in range(pre_attack_start, attack_end_sample):
                        attack_indices_flat.add(i)
                
                non_attack_indices = [i for i in selected_indices if i not in attack_indices_flat]
                
                # Calculate how many non-attack points to keep
                to_keep = test_sample_size - len(attack_indices_flat)
                if to_keep > 0:
                    # Sample evenly from non-attack points
                    if len(non_attack_indices) > to_keep:
                        keep_indices = np.linspace(0, len(non_attack_indices)-1, to_keep, dtype=int)
                        non_attack_to_keep = [non_attack_indices[i] for i in keep_indices]
                    else:
                        non_attack_to_keep = non_attack_indices
                    
                    # Combine attack points with sampled non-attack points
                    selected_indices = list(attack_indices_flat) + non_attack_to_keep
                else:
                    # If we have too many attack points, subsample from them
                    selected_indices = list(attack_indices_flat)[:test_sample_size]
            
            # Sort indices to maintain temporal order
            selected_indices = sorted(selected_indices)
            
            # Apply sampling
            test_data = test_data.iloc[selected_indices].reset_index(drop=True)
            
            print(f"Sampled data: train={len(initial_train_data)} rows (regular sampling), test={len(test_data)} rows (attack-focused)")
        else:
            # Use original sampling method for test data
            test_sample_size = int(len(test_data) * sample_rate)
            test_indices = np.linspace(0, len(test_data)-1, test_sample_size, dtype=int)
            test_data = test_data.iloc[test_indices].reset_index(drop=True)
            print(f"Sampled data: train={len(initial_train_data)} rows, test={len(test_data)} rows (both using regular sampling)")
    else:
        print(f"Using full dataset: train={len(initial_train_data)} rows, test={len(test_data)} rows")

    # --- Manual Train/Validation Split ---
    if validation_split_ratio > 0 and len(initial_train_data) > 1:
        # Calculate split point based on the *current* length (after potential sampling)
        split_idx = int(len(initial_train_data) * (1 - validation_split_ratio))

        # Get indices *without* shuffling to preserve order from linspace sampling
        # This ensures validation set maintains the sequence if sample_rate < 1.0
        all_indices = initial_train_data.index.to_numpy() # Use current index after sampling
        train_indices = all_indices[:split_idx]
        validation_indices = all_indices[split_idx:]
        print(f"Splitting train data sequentially (no shuffle) for validation.") # Add print statement
        
        # Create the dataframes
        train_data = initial_train_data.iloc[train_indices].reset_index(drop=True)
        validation_data = initial_train_data.iloc[validation_indices].reset_index(drop=True)
        
        print(f"Manually split training data: final train={len(train_data)} rows, validation={len(validation_data)} rows")
    else:
        # No validation split requested or not enough data
        train_data = initial_train_data
        validation_data = pd.DataFrame() # Empty DataFrame for consistency
        if validation_split_ratio > 0:
             print("Warning: Not enough data in initial_train_data to perform validation split.")
        else:
             print("No validation split performed.")

    # Convert 'Attack' column to binary (WADI uses 'Attack' not 'Normal/Attack')
    if 'Attack' in train_data.columns:
        train_data['Normal/Attack'] = train_data['Attack'].map(lambda x: 0 if x == 0 else 1)
        if not validation_data.empty:
            validation_data['Normal/Attack'] = validation_data['Attack'].map(lambda x: 0 if x == 0 else 1)
        test_data['Normal/Attack'] = test_data['Attack'].map(lambda x: 0 if x == 0 else 1)
    else:
        # Handle case where it might already be 'Normal/Attack'
        train_data['Normal/Attack'] = train_data['Normal/Attack'].map(lambda x: 0 if (x == 'False' or x == 0 or x == 0.0 or x == 'Normal') else 1)
        if not validation_data.empty:
            validation_data['Normal/Attack'] = validation_data['Normal/Attack'].map(lambda x: 0 if (x == 'False' or x == 0 or x == 0.0 or x == 'Normal') else 1)
        test_data['Normal/Attack'] = test_data['Normal/Attack'].map(lambda x: 0 if (x == 'False' or x == 0 or x == 0.0 or x == 'Normal') else 1)
    
    # Check label distribution and print it for verification
    normal_test = (test_data['Normal/Attack'] == 0).sum()
    attack_test = (test_data['Normal/Attack'] == 1).sum()
    print(f"Test data contains {normal_test} normal samples and {attack_test} attack samples")
    print(f"Attack percentage: {attack_test/len(test_data)*100:.2f}%")

    # Save test data if path provided
    if save_test_path:
        print(f"Saving test data to {save_test_path}...")
        os.makedirs(os.path.dirname(save_test_path), exist_ok=True)
        
        # Determine file extension and save accordingly
        if save_test_path.endswith('.csv'):
            test_data.to_csv(save_test_path, index=False)
        else:
            with open(save_test_path, 'wb') as f:
                pickle.dump(test_data, f)
        print(f"Test data saved successfully with {len(test_data)} samples")

    return train_data, validation_data, test_data

def load_saved_test_data(file_path):
    """
    Load previously saved test data.
    
    Parameters
    ----------
    file_path : str
        Path to the saved test data file
        
    Returns
    -------
    pandas.DataFrame
        The test data
    """
    print(f"Loading test data from {file_path}...")
    with open(file_path, 'rb') as f:
        test_data = pickle.load(f)
    
    print(f"Loaded test data with {len(test_data)} samples")
    
    # Print label distribution
    normal_test = (test_data['Normal/Attack'] == 0.0).sum()
    attack_test = (test_data['Normal/Attack'] == 1.0).sum()
    print(f"Test data contains {normal_test} normal samples and {attack_test} attack samples")
    print(f"Attack percentage: {attack_test/len(test_data)*100:.2f}%")
    
    return test_data

def run_experiment(config):
    """Main function to run WADI experiment based on a config dictionary."""
    print("Starting WADI Anomaly Detection Experiment (3-level)...")

    # Set device from config
    device = config['system']['device']
    print(f"Using device: {device}")

    # Define paths from config
    data_dir = config['data']['data_dir']
    train_path = os.path.join(data_dir, config['data']['train_path'])
    test_path = os.path.join(data_dir, config['data']['test_path'])
    
    # Create directory for saved test data
    saved_data_dir = "saved_test_data"
    os.makedirs(saved_data_dir, exist_ok=True)

    # Load or create test data
    saved_test_path = os.path.join(saved_data_dir, f"wadi_test_data_{config['data']['sample_rate']}.pkl")
    
    if os.path.exists(saved_test_path):
        print(f"Loading pre-sampled test data from {saved_test_path}")
        test_data = load_saved_test_data(saved_test_path)

        # Since test data is loaded, we only need to load train and validation sets.
        print("Loading corresponding training and validation data...")
        train_data, validation_data, _ = load_wadi_data(
            train_path=train_path,
            test_path=test_path,  # Still required by the function, will be loaded and discarded
            sample_rate=config['data']['sample_rate'],
            save_test_path=None,  # Do not save again
            validation_split_ratio=config['data']['validation_split_ratio']
        )
    else:
        print("Creating new train/validation/test split and saving test data...")
        train_data, validation_data, test_data = load_wadi_data(
            train_path=train_path,
            test_path=test_path,
            sample_rate=config['data']['sample_rate'],
            save_test_path=saved_test_path, # Save for next time
            validation_split_ratio=config['data']['validation_split_ratio']
        )

    # Load WADI topology
    print("Loading WADI topology...")
    
    # Get component names for WADI topology - use same exclude list as WADIDataset
    exclude_columns = ['Timestamp', 'Normal/Attack', 'Attack', 'Row', 'Date', 'Time', 
                      '2B_AIT_002_PV', '2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', 
                      '2_P_002_STATUS', 'LEAK_DIFF_PRESSURE', 'PLANT_START_STOP_LOG', 
                      'TOTAL_CONS_REQUIRED_FLOW']
    component_names = [col for col in train_data.columns if col not in exclude_columns]
    print(f"DEBUG: Building WADI topology with {len(component_names)} components")
    wadi_complex = WADIComplex(component_names)

    temporal_mode = config['data']['temporal_mode']

    # Create datasets
    print(f"Creating datasets (train, validation, test) in {'temporal' if temporal_mode else 'reconstruction'} mode...")
    if temporal_mode:
        dataset_args = {
            'temporal_mode': True,
            'n_input': config['model']['n_input'],
            'temporal_sample_rate': config['data']['temporal_sample_rate']
        }
    else:
        dataset_args = {'temporal_mode': False}

    train_dataset = WADIDataset(train_data, wadi_complex, **dataset_args)
    validation_dataset = WADIDataset(validation_data, wadi_complex, **dataset_args) if not validation_data.empty else None
    test_dataset = WADIDataset(test_data, wadi_complex, **dataset_args)

    # Create dataloaders
    batch_size = config['training']['batch_size']
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if validation_dataset:
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    else:
        validation_dataloader = None
        print("Warning: No validation data available. Threshold will be calibrated on training data.")

    if validation_dataloader is None:
         print("Error: Validation dataloader could not be created. Check validation_split_ratio.")
         return

    # Create model
    print(f"Creating model in {'temporal' if temporal_mode else 'reconstruction'} mode...")
    
    # Get feature dimensions from the dataset
    original_feature_dims = {
        '0': train_dataset.feature_dim_0,
        '1': train_dataset.feature_dim_1,
        '2': train_dataset.feature_dim_2
    }
    print(f"Using feature dimensions: {original_feature_dims}")
    
    model = AnomalyCCANN(
        config['model']['channels_per_layer'], 
        original_feature_dims=original_feature_dims,
        temporal_mode=temporal_mode,
        n_input=config['model'].get('n_input', 10)
    )

    # Create trainer with validation dataloader
    print("Creating trainer with validation set for thresholding...")
    
    # Process evaluation config
    eval_config = config.get('evaluation', {})
    if 'method' in eval_config:
        eval_config['evaluation_method'] = eval_config.pop('method')
    if 'fp_alarm_window_seconds' in eval_config:
        eval_config['fp_alarm_window'] = eval_config.pop('fp_alarm_window_seconds')
    
    # Remove threshold_percentile from eval_config to avoid duplicate parameter
    if 'threshold_percentile' in eval_config:
        eval_config.pop('threshold_percentile')
    
    # Add enhanced time-aware metrics parameters
    enhanced_metrics_config = config.get('enhanced_time_aware_metrics', {})
    eval_config.update({
        'theta_p': enhanced_metrics_config.get('theta_p', 0.5),
        'theta_r': enhanced_metrics_config.get('theta_r', 0.1),
        'original_sample_hz': enhanced_metrics_config.get('original_sample_hz', 1)
    })
        
    trainer = AnomalyTrainer(
        model,
        train_dataloader,
        validation_dataloader,
        test_dataloader,
        learning_rate=float(config['training']['learning_rate']),
        device=device,
        weight_decay=float(config['training']['weight_decay']),
        grad_clip_value=float(config['training']['grad_clip_value']),
        # Pass sample rate for proper attack index adjustment
        sample_rate=float(config['data']['sample_rate']),
        # Pass dataset name for attack indices
        dataset_name=config.get('dataset_name', 'WADI'),
        # Pass custom level names if specified
        level_names=config.get('level_names', None),
        # Pass anomaly detection config
        temporal_consistency=config.get('anomaly_detection', {}).get('temporal_consistency', 1),
        threshold_method=config.get('anomaly_detection', {}).get('threshold_method', 'percentile'),
        threshold_percentile=config.get('anomaly_detection', {}).get('threshold_percentile', 99.0),
        sd_multiplier=config.get('anomaly_detection', {}).get('sd_multiplier', 2.5),
        use_component_thresholds=config.get('anomaly_detection', {}).get('use_component_thresholds', False),
        # Pass the corrected evaluation config block
        **eval_config
    )

    # Create checkpoint directory
    checkpoint_dir = f"{config['system']['checkpoint_dir']}/{config['experiment_name']}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train model with early stopping
    final_test_results = trainer.train(
        num_epochs=int(config['training']['epochs']),
        test_interval=int(config['training']['test_interval']), 
        checkpoint_dir=checkpoint_dir,
        early_stopping=config['training']['early_stopping']['enabled'],
        patience=int(config['training']['early_stopping']['patience']),
        min_delta=float(config['training']['early_stopping']['min_delta'])
    )

    # Save final test results
    results_path = os.path.join(checkpoint_dir, "final_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(final_test_results, f)
    print(f"Final results saved to {results_path}")
    
    print("Experiment finished successfully!") 