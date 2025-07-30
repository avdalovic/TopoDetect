import os
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch
import random

# Imports assuming the new structure
from src.utils.topology.wadi_topology import WADIComplex
from src.utils.attack_utils import get_attack_indices, get_attack_sds
from src.datasets.wadi_dataset import WADIDataset
from src.models.ccann import AnomalyCCANN
from src.trainers.anomaly_trainer import AnomalyTrainer


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    print(f"Setting random seed to {seed} for reproducibility")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_wadi_data(train_path, test_path, sample_rate=1.0, save_test_path=None, validation_split_ratio=0.2, 
                  attack_focused_sampling=False, pre_attack_points=10, attack_start_points=5, 
                  remove_stabilization_points=0):
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
    remove_stabilization_points : int, default=0
        Number of initial data points to remove from training data for stabilization.
        Based on WADI analysis, 0 is recommended (WADI is stable from start).
        Use values like 3600 (1h) or 7200 (2h) only for experimentation.

    Returns
    -------
    tuple of pandas.DataFrame
        (train_data, validation_data, test_data)
    """
    print(f"Loading WADI data from {train_path} and {test_path}...")
    print(f"Using sample rate: {sample_rate}")
    print(f"Using validation split ratio: {validation_split_ratio}")
    print(f"Remove stabilization points: {remove_stabilization_points}")

    # Load data
    initial_train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Optional stabilization removal for WADI (default: no removal based on analysis)
    if remove_stabilization_points > 0:
        if len(initial_train_data) > remove_stabilization_points:
            print(f"Removing first {remove_stabilization_points} data points for WADI stabilization (experimental)")
            initial_train_data = initial_train_data.iloc[remove_stabilization_points:].reset_index(drop=True)
            print(f"Training data after stabilization removal: {len(initial_train_data)} samples")
        else:
            print(f"Warning: Training data has only {len(initial_train_data)} samples, cannot remove {remove_stabilization_points} points")
    else:
        print("WADI data is stable from the start - no stabilization removal needed (recommended)")

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
    # --- End Manual Split ---

    # Convert 'Attack' to boolean/int for consistency
    train_data['Attack'] = train_data['Attack'].astype(int)
    if not validation_data.empty:
        validation_data['Attack'] = validation_data['Attack'].astype(int)
    # Ensure test data labels are also 0 or 1
    test_data['Attack'] = test_data['Attack'].astype(int)
    
    # Check label distribution and print it for verification
    normal_test = (test_data['Attack'] == 0).sum()
    attack_test = (test_data['Attack'] == 1).sum()
    print(f"Test data contains {normal_test} normal samples and {attack_test} attack samples")
    print(f"Attack percentage: {attack_test/len(test_data)*100:.2f}%")
    
    # Additional debug info for WADI
    print(f"DEBUG: WADI test data analysis:")
    print(f"  Sample rate used: {sample_rate}")
    print(f"  Total test samples: {len(test_data)}")
    print(f"  Attack samples: {attack_test}")
    if attack_test == 0:
        print(f"  WARNING: No attack samples in test data! This will cause evaluation to fail.")
        print(f"  Consider increasing sample_rate or using attack_focused_sampling=True")

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
    normal_test = (test_data['Attack'] == 0.0).sum()
    attack_test = (test_data['Attack'] == 1.0).sum()
    print(f"Test data contains {normal_test} normal samples and {attack_test} attack samples")
    print(f"Attack percentage: {attack_test/len(test_data)*100:.2f}%")
    
    return test_data

def run_experiment(config):
    """Main function to run WADI experiment based on a config dictionary."""
    print("Starting WADI Anomaly Detection Experiment (3-level)...")
    
    # Set seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)

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
        # We call load_wadi_data but ignore its test_data output and prevent it from saving.
        print("Loading corresponding training and validation data...")
        train_data, validation_data, _ = load_wadi_data(
            train_path=train_path,
            test_path=test_path,  # Still required by the function, will be loaded and discarded
            sample_rate=config['data']['sample_rate'],
            save_test_path=None,  # Do not save again
            validation_split_ratio=config['data']['validation_split_ratio'],
            remove_stabilization_points=config['data'].get('remove_stabilization_points', 0)
        )
    else:
        print("Creating new train/validation/test split and saving test data...")
        train_data, validation_data, test_data = load_wadi_data(
            train_path=train_path,
            test_path=test_path,
            sample_rate=config['data']['sample_rate'],
            save_test_path=saved_test_path, # Save for next time
            validation_split_ratio=config['data']['validation_split_ratio'],
            remove_stabilization_points=config['data'].get('remove_stabilization_points', 0)
        )

    # Create WADI topology complex
    print("Creating WADI topology complex...")
    
    # Get filtered component names (after removing problematic features)
    # We'll create a temporary dataset just to get the filtered component names
    temp_data = train_data.copy()
    exclude_columns = ['Row', 'Date', 'Time', 'Attack', 'Normal/Attack']
    all_columns = [col for col in temp_data.columns if col not in exclude_columns]
    
    # Remove problematic features
    remove_list = [
        '2B_AIT_002_PV', '2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS',
        'LEAK_DIFF_PRESSURE', 'PLANT_START_STOP_LOG', 'TOTAL_CONS_REQUIRED_FLOW'
    ]
    
    filtered_components = [col for col in all_columns if col not in remove_list]
    print(f"Creating WADI complex with {len(filtered_components)} components (after filtering)")
    
    # Create WADI topology with optional GECO features and pure embedding support
    use_geco_relationships = config.get('topology', {}).get('use_geco_relationships', True)
    use_pure_embedding = config.get('topology', {}).get('use_pure_embedding', False)
    
    if use_pure_embedding:
        print("🧠 Using Pure Embedding Topology (NO GECO)")
        from src.utils.topology.pure_embedding_topology import create_pure_embedding_wadi_complex
        
        embeddings_file = config.get('embedding', {}).get('embeddings_file', 'embeddings/wadi_embeddings.pkl')
        similarity_threshold = config.get('topology', {}).get('embedding_similarity_threshold', 0.3)
        min_edges_per_node = config.get('topology', {}).get('min_edges_per_node', 3)
        
        print(f"  📊 Embeddings file: {embeddings_file}")
        print(f"  🎯 Similarity threshold: {similarity_threshold}")
        print(f"  🔗 Min edges per node: {min_edges_per_node}")
        
        pure_complex = create_pure_embedding_wadi_complex(
            embeddings_file=embeddings_file,
            similarity_threshold=similarity_threshold,
            min_edges_per_node=min_edges_per_node
        )
        
        # Create a wrapper object to match WADIDataset expectations
        class PureEmbeddingWrapper:
            def __init__(self, complex):
                self._complex = complex
            def get_complex(self):
                return self._complex
            def _get_node_index_map(self):
                """Helper to create a mapping from component name to its index in the complex."""
                row_dict_01, _, _ = self._complex.incidence_matrix(0, 1, index=True)
                return {list(k)[0]: v for k, v in row_dict_01.items()}
        
        wadi_complex = PureEmbeddingWrapper(pure_complex)
        complex_obj = wadi_complex.get_complex()
    else:
        wadi_complex = WADIComplex(component_names=filtered_components, use_geco_relationships=use_geco_relationships)
        complex_obj = wadi_complex.get_complex()
        
        if use_geco_relationships:
            print("Using GECO-learned relationships from WADI.model")
        else:
            print("Using manually defined relationships")
    
    print(f"WADI complex created: {complex_obj}")
    
    # Create datasets
    print("Creating datasets (train, validation, test) in reconstruction mode...")
    
    # Get new feature engineering parameters
    normalization_method = config.get('data', {}).get('normalization_method', 'standard')
    use_enhanced_2cell_features = config.get('data', {}).get('use_enhanced_2cell_features', False)
    use_first_order_differences = config.get('data', {}).get('use_first_order_differences', False)
    use_first_order_differences_edges = config.get('data', {}).get('use_first_order_differences_edges', True)
    use_pressure_differential_features = config.get('data', {}).get('use_pressure_differential_features', False)
    
    print(f"Feature engineering config:")
    print(f"  normalization={normalization_method}")
    print(f"  enhanced_2cell={use_enhanced_2cell_features}")
    print(f"  first_order_diff_nodes={use_first_order_differences}")
    print(f"  first_order_diff_edges={use_first_order_differences_edges}")
    print(f"  pressure_differential_features={use_pressure_differential_features} (DEPRECATED - causes loss instability)")
    
    # Check if GECO features should be used
    use_geco_features = config.get('topology', {}).get('use_geco_features', False)
    
    # OPTIMIZATION: For proper MinMax normalization, create training dataset first
    if normalization_method == "minmax_proper":
        print("Using proper MinMax normalization - creating training dataset first to establish parameters...")
        
        # Create training dataset first
        train_dataset = WADIDataset(
            data=train_data,
            wadi_complex=wadi_complex,
            temporal_mode=config['data']['temporal_mode'],
            temporal_sample_rate=config['data']['temporal_sample_rate'],
            use_geco_features=use_geco_features,
            normalization_method=normalization_method,
            use_enhanced_2cell_features=use_enhanced_2cell_features,
            use_first_order_differences=use_first_order_differences,
            use_first_order_differences_edges=use_first_order_differences_edges,
            use_pressure_differential_features=use_pressure_differential_features,
            use_categorical_embeddings=config.get('data', {}).get('use_categorical_embeddings', False),
            seed=config.get('seed', 42),
            is_training_data=True  # Training data gets stabilization removal
        )
        
        # Get expected feature count from training dataset
        expected_feature_count = len(train_dataset.columns)
        print(f"Training dataset established {expected_feature_count} features as the standard")
        
        # Create validation dataset with shared parameters
        if not validation_data.empty:
            print("Creating validation dataset with shared training parameters...")
            validation_dataset = WADIDataset(
                data=validation_data,
                wadi_complex=wadi_complex,
                temporal_mode=config['data']['temporal_mode'],
                temporal_sample_rate=config['data']['temporal_sample_rate'],
                use_geco_features=use_geco_features,
                normalization_method=normalization_method,
                use_enhanced_2cell_features=use_enhanced_2cell_features,
                use_first_order_differences=use_first_order_differences,
                use_first_order_differences_edges=use_first_order_differences_edges,
                use_pressure_differential_features=use_pressure_differential_features,
                use_categorical_embeddings=config.get('data', {}).get('use_categorical_embeddings', False),
                seed=config.get('seed', 42),
                is_training_data=False  # Validation data gets no stabilization removal
            )
            
            # Share training parameters
            validation_dataset.train_min_vals = train_dataset.train_min_vals
            validation_dataset.train_max_vals = train_dataset.train_max_vals
            validation_dataset.expected_feature_count = expected_feature_count
            
            # Recompute features with shared parameters
            # Ensure validation dataset uses the same method as training dataset
            validation_dataset.use_categorical_embeddings = train_dataset.use_categorical_embeddings
            validation_dataset.x_0 = validation_dataset.compute_initial_x0()
            validation_dataset.x_1 = validation_dataset.compute_initial_x1()
            validation_dataset.x_2 = validation_dataset.compute_initial_x2()
            
            print(f"Validation dataset created with {len(validation_dataset)} samples")
        else:
            validation_dataset = None
            print("No validation data available")
        
        # Create test dataset with shared parameters
        print("Creating test dataset with shared training parameters...")
        test_dataset = WADIDataset(
            data=test_data,
            wadi_complex=wadi_complex,
            temporal_mode=config['data']['temporal_mode'],
            temporal_sample_rate=config['data']['temporal_sample_rate'],
            use_geco_features=use_geco_features,
            normalization_method=normalization_method,
            use_enhanced_2cell_features=use_enhanced_2cell_features,
            use_first_order_differences=use_first_order_differences,
            use_first_order_differences_edges=use_first_order_differences_edges,
            use_pressure_differential_features=use_pressure_differential_features,
            use_categorical_embeddings=config.get('data', {}).get('use_categorical_embeddings', False),
            seed=config.get('seed', 42),
            is_training_data=False  # Test data gets no stabilization removal
        )
        
        # Share training parameters
        test_dataset.train_min_vals = train_dataset.train_min_vals
        test_dataset.train_max_vals = train_dataset.train_max_vals
        test_dataset.expected_feature_count = expected_feature_count
        
        # Recompute features with shared parameters
        # Ensure test dataset uses the same method as training dataset
        test_dataset.use_categorical_embeddings = train_dataset.use_categorical_embeddings
        test_dataset.x_0 = test_dataset.compute_initial_x0()
        test_dataset.x_1 = test_dataset.compute_initial_x1()
        test_dataset.x_2 = test_dataset.compute_initial_x2()
        
        print(f"Test dataset created with {len(test_dataset)} samples")
        
    else:
        # Standard approach for other normalization methods
        print("Using standard dataset creation approach...")
        
        train_dataset = WADIDataset(
            data=train_data,
            wadi_complex=wadi_complex,
            temporal_mode=config['data']['temporal_mode'],
            temporal_sample_rate=config['data']['temporal_sample_rate'],
            use_geco_features=use_geco_features,
            normalization_method=normalization_method,
            use_enhanced_2cell_features=use_enhanced_2cell_features,
            use_first_order_differences=use_first_order_differences,
            use_first_order_differences_edges=use_first_order_differences_edges,
            use_pressure_differential_features=use_pressure_differential_features,
            use_categorical_embeddings=config.get('data', {}).get('use_categorical_embeddings', False),
            seed=config.get('seed', 42),
            is_training_data=True  # Training data gets stabilization removal
        )
        
        # Get expected feature count from training dataset
        expected_feature_count = len(train_dataset.columns)
        print(f"Training dataset established {expected_feature_count} features as the standard")
        
        validation_dataset = WADIDataset(
            data=validation_data,
            wadi_complex=wadi_complex,
            temporal_mode=config['data']['temporal_mode'],
            temporal_sample_rate=config['data']['temporal_sample_rate'],
            use_geco_features=use_geco_features,
            normalization_method=normalization_method,
            use_enhanced_2cell_features=use_enhanced_2cell_features,
            use_first_order_differences=use_first_order_differences,
            use_first_order_differences_edges=use_first_order_differences_edges,
            use_pressure_differential_features=use_pressure_differential_features,
            use_categorical_embeddings=config.get('data', {}).get('use_categorical_embeddings', False),
            seed=config.get('seed', 42),
            is_training_data=False  # Validation data gets no stabilization removal
        ) if not validation_data.empty else None
        
        # Ensure validation dataset matches training feature count (only if dataset exists)
        if validation_dataset is not None:
            validation_dataset.expected_feature_count = expected_feature_count
        
        test_dataset = WADIDataset(
            data=test_data,
            wadi_complex=wadi_complex,
            temporal_mode=config['data']['temporal_mode'],
            temporal_sample_rate=config['data']['temporal_sample_rate'],
            use_geco_features=use_geco_features,
            normalization_method=normalization_method,
            use_enhanced_2cell_features=use_enhanced_2cell_features,
            use_first_order_differences=use_first_order_differences,
            use_first_order_differences_edges=use_first_order_differences_edges,
            use_pressure_differential_features=use_pressure_differential_features,
            use_categorical_embeddings=config.get('data', {}).get('use_categorical_embeddings', False),
            seed=config.get('seed', 42),
            is_training_data=False  # Test data gets no stabilization removal
        )
        # Ensure test dataset matches training feature count
        test_dataset.expected_feature_count = expected_feature_count

    # Create dataloaders
    batch_size = config['training']['batch_size']
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False) # Turn off pin_memory
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if validation_dataset is not None:
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        print(f"Created validation dataloader with {len(validation_dataset)} samples")
    else:
        validation_dataloader = None
        print("Warning: No validation data available. Threshold will be calibrated on training data.")
        if config['anomaly_detection']['threshold_method'] != 'train':
            print("Switching to 'train' thresholding method.")
            config['anomaly_detection']['threshold_method'] = 'train'

    # Remove this check since we can proceed without validation data
    # if validation_dataloader is None:
    #      print("Error: Validation dataloader could not be created. Check validation_split_ratio.")
    #      return

    # Create model
    print(f"Creating model in {'temporal' if config['data']['temporal_mode'] else 'reconstruction'} mode...")
    
    # Get feature dimensions from the dataset
    original_feature_dims = {
        '0': train_dataset.feature_dim_0,
        '1': train_dataset.feature_dim_1,
        '2': train_dataset.feature_dim_2
    }
    print(f"Using feature dimensions: {original_feature_dims}")
    
    if config['data'].get('use_geco_features', False):
        print("GECO features are enabled in the dataset.")
    else:
        print("GECO features are disabled.")
    
    # Dynamically update model input channels based on dataset feature dimensions
    # This ensures the model architecture from the config is respected while using correct input dimensions
    channels_per_layer = config['model']['channels_per_layer']
    
    # Check if categorical embeddings are enabled
    use_categorical_embeddings = config['model'].get('use_categorical_embeddings', False)
    categorical_embedding_dim = config['model'].get('categorical_embedding_dim', 8)
    
    # Calculate input dimensions
    base_dim_0 = train_dataset.feature_dim_0
    if use_categorical_embeddings:
        # Categorical embeddings enhance 0-cell features
        enhanced_dim_0 = base_dim_0 + categorical_embedding_dim
        print(f"Categorical embeddings enabled: enhancing 0-cell features from {base_dim_0}D to {enhanced_dim_0}D")
    else:
        enhanced_dim_0 = base_dim_0
    
    in_channels = [
        enhanced_dim_0,  # 0-cells: base_dim + embeddings (if enabled)
        train_dataset.feature_dim_1,  # 1-cells: unchanged
        train_dataset.feature_dim_2   # 2-cells: unchanged
    ]
    channels_per_layer[0][0] = in_channels
    print(f"Dynamically updated model input channels to: {in_channels}")

    model = AnomalyCCANN(
        channels_per_layer, 
        original_feature_dims=original_feature_dims,
        temporal_mode=config['data']['temporal_mode'],
        n_input=config['model'].get('n_input', 10), # Get n_input safely
        use_enhanced_decoders=config['model'].get('use_enhanced_decoders', False),
        use_categorical_embeddings=config['model'].get('use_categorical_embeddings', False),
        num_sensor_types=config['model'].get('num_sensor_types', 12),
        categorical_embedding_dim=config['model'].get('categorical_embedding_dim', 8)
    )

    # Ensure model is properly initialized on device
    print(f"Model created. Moving to device: {device}")
    model = model.to(device)

    # Create trainer with validation dataloader
    print("Creating trainer with validation set for thresholding...")
    
    # Ensure proper device initialization
    if str(device).startswith('cuda'):
        import torch.cuda
        if not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
        else:
            # Initialize CUDA context
            torch.cuda.empty_cache()
            print(f"CUDA initialized in pipeline. Using device: {device}")
    
    # Correction: Rename 'method' key from config to 'evaluation_method' for trainer
    eval_config = config.get('evaluation', {})
    if 'method' in eval_config:
        eval_config['evaluation_method'] = eval_config.pop('method')
    # Correction: Rename 'fp_alarm_window_seconds' to 'fp_alarm_window'
    if 'fp_alarm_window_seconds' in eval_config:
        eval_config['fp_alarm_window'] = eval_config.pop('fp_alarm_window_seconds')
    
    # Remove parameters that AnomalyTrainer doesn't accept
    for param in ['threshold_percentile', 'metrics', 'save_predictions', 'save_reconstruction_errors']:
        if param in eval_config:
            eval_config.pop(param)
    
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
        use_two_threshold_alarm=config.get('anomaly_detection', {}).get('use_two_threshold_alarm', False),
        plc_low_percentile=config.get('anomaly_detection', {}).get('plc_low_percentile', 99.0),
        plc_high_percentile=config.get('anomaly_detection', {}).get('plc_high_percentile', 99.9),

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