import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.utils.attack_utils import is_actuator

def set_seed(seed):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class WADIDataset(Dataset):
    """WADI dataset for topological deep learning with combinatorial complexes."""
    
    def __init__(self, data, wadi_complex, feature_dim_0=3, n_input=10, temporal_mode=False, temporal_sample_rate=1, use_geco_features=False, 
                 normalization_method="standard", use_enhanced_2cell_features=False, use_first_order_differences=False,
                 use_first_order_differences_edges=True, use_pressure_differential_features=False, seed=None):
        """
        Initialize WADI dataset.
        
        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing the WADI measurements and labels
        wadi_complex : WADIComplex
            Complex representing the WADI system topology
        feature_dim_0 : int, default=3
            Base dimension of feature vectors for 0-cells
        n_input : int, default=10
            Number of input timesteps for temporal mode
        temporal_mode : bool, default=False
            Whether to use temporal mode
        temporal_sample_rate : int, default=1
            Sampling rate for temporal mode
        use_geco_features : bool, default=False
            Whether to use enhanced edge features with GECO information
        normalization_method : str, default="standard"
            Normalization method: "standard", "z_normalization", "mixed_normalization"
        use_enhanced_2cell_features : bool, default=False
            Whether to use enhanced 2-cell features (4D)
        use_first_order_differences : bool, default=False
            Whether to include first-order differences (lag-1) as additional features for 0-cells
        use_first_order_differences_edges : bool, default=True
            Whether to include first-order differences for 1-cells (edges) - separate control
        use_pressure_differential_features : bool, default=False
            DEPRECATED: Physics features removed due to causing loss instability
        seed : int, optional
            Random seed for reproducibility
        """
        self.data = data
        self.wadi_complex = wadi_complex
        self.complex = wadi_complex.get_complex()
        
        # Remove first 21600 points (6 hours) from training data for WADI stabilization (like SWAT)
        stabilization_point = 21600
        if len(self.data) > stabilization_point:
            print(f"Removing first {stabilization_point} data points (6 hours) for WADI stabilization")
            self.data = self.data.iloc[stabilization_point:].reset_index(drop=True)
            print(f"Training data after stabilization removal: {len(self.data)} samples")
        else:
            print(f"Warning: Training data has only {len(self.data)} samples, cannot remove {stabilization_point} points")
        self.feature_dim_0 = feature_dim_0
        self.n_input = n_input
        self.temporal_mode = temporal_mode
        self.temporal_sample_rate = temporal_sample_rate
        self.use_geco_features = use_geco_features
        self.normalization_method = normalization_method
        self.use_enhanced_2cell_features = use_enhanced_2cell_features
        self.use_first_order_differences = use_first_order_differences
        self.use_first_order_differences_edges = use_first_order_differences_edges
        self.use_pressure_differential_features = use_pressure_differential_features
        self.seed = seed
        
        # Set seed if provided
        if seed is not None:
            set_seed(seed)
        
        print(f"Using GECO features: {use_geco_features}")
        print(f"Using normalization method: {normalization_method}")
        print(f"Using enhanced 2-cell features: {use_enhanced_2cell_features}")
        print(f"Using first-order differences (0-cells): {use_first_order_differences}")
        print(f"Using first-order differences (1-cells): {use_first_order_differences_edges}")
        if seed is not None:
            print(f"Using seed: {seed}")
        
        # Set feature dimensions based on configuration
        # Determine actual 0-cell feature dimension based on normalization method and first-order differences
        if normalization_method == "standard":
            base_feature_dim_0 = 3  # Original 3D features
        else:
            base_feature_dim_0 = 1  # New methods use 1D features
            
        # Add first-order differences dimension if enabled for 0-cells
        if self.use_first_order_differences:
            actual_feature_dim_0 = base_feature_dim_0 + 1  # Add 1D for first-order difference
            print(f"0-cell feature dimension increased from {base_feature_dim_0}D to {actual_feature_dim_0}D (added first-order differences)")
        else:
            actual_feature_dim_0 = base_feature_dim_0
            
        # Calculate 1-cell feature dimensions
        # For edges, we only use the FIRST dimension of 0-cells (normalized values, not differences)
        if self.use_geco_features:
            if actual_feature_dim_0 == 3:  # Original implementation
                base_1cell_dim = actual_feature_dim_0 * 2 + 1  # 6D + 1D GECO = 7D
            else:  # New implementation with 1D or 2D 0-cells
                # We only use the first dimension (normalized values) from each 0-cell for edges
                base_1cell_dim = 1 * 2 + 1  # 2 * 1D values + 1D GECO = 3D
        else:
            if actual_feature_dim_0 == 3:  # Original implementation
                base_1cell_dim = actual_feature_dim_0 * 2  # 6D concatenated features
            else:  # New implementation with 1D or 2D 0-cells
                # We only use the first dimension (normalized values) from each 0-cell for edges
                base_1cell_dim = 1 * 2  # 2 * 1D values = 2D
        
        # Add first-order differences for 1-cells if enabled
        if self.use_first_order_differences_edges:
            self.feature_dim_1 = base_1cell_dim + 1  # Add 1D for edge first-order differences
            print(f"Using 1-cell features with differences: {self.feature_dim_1}D")
        else:
            self.feature_dim_1 = base_1cell_dim
            print(f"Using 1-cell features without differences: {self.feature_dim_1}D")
        
        # Note: Cosine similarity will be added dynamically in _compute_new_x1 if available
        
        # Override the feature_dim_0 for consistency
        self.feature_dim_0 = actual_feature_dim_0
        
        # Set 2-cell feature dimensions
        base_2cell_dim = 4 if use_enhanced_2cell_features else 1
        self.feature_dim_2 = base_2cell_dim  # Removed physics feature dimension
        print(f"Using standard 2-cell features: {self.feature_dim_2}D")

        # Preprocess the data
        self.preprocess_features()
        
        # Create component to index mapping
        self.component_to_idx = {comp: i for i, comp in enumerate(self.columns)}
        
        # Compute features based on normalization method
        if normalization_method == "standard":
            self.x_0 = self.compute_initial_x0()
        elif normalization_method == "z_normalization":
            self.x_0 = self._compute_z_normalized_x0()
        elif normalization_method == "mixed_normalization":
            self.x_0 = self._compute_mixed_normalized_x0()
        elif normalization_method == "median_subtraction":
            self.x_0 = self._compute_median_subtraction_x0()
        elif normalization_method == "minmax":
            self.x_0 = self._compute_minmax_x0()
        elif normalization_method == "minmax_proper":
            self.x_0 = self._compute_minmax_proper_x0()
        else:
            raise ValueError(f"Unknown normalization method: {normalization_method}")
        
        # Compute 1-cell and 2-cell features
        if use_enhanced_2cell_features:
            self.x_1 = self._compute_new_x1()  # 3D features
            self.x_2 = self._compute_enhanced_x2()  # 4D features
        else:
            self.x_1 = self.compute_initial_x1()
            self.x_2 = self.compute_initial_x2()
        
        # Extract labels
        if 'Attack' in self.data.columns:
            self.labels = self.data['Attack'].values
        elif 'Normal/Attack' in self.data.columns:
            self.labels = self.data['Normal/Attack'].values
        else:
            # If no label column, assume all normal (for some test scenarios)
            self.labels = np.zeros(len(self.data))
        
        # Create adjacency matrices
        self.create_adjacency_matrices()
        
        # Create temporal samples if needed
        if self.temporal_mode:
            self.create_temporal_samples()
        else:
            self.effective_length = len(self.data)
        
        print(f"WADIDataset initialized with {len(self)} samples")

    def preprocess_features(self):
        """
        Extract and preprocess features from the data.
        """
        # Extract features and labels
        if 'Attack' in self.data.columns:
            self.labels = self.data['Attack'].values
        elif 'Normal/Attack' in self.data.columns:
            self.labels = self.data['Normal/Attack'].values
        else:
            # If no label column, assume all normal (for some test scenarios)
            self.labels = np.zeros(len(self.data))

        # Get sensor and actuator columns (excluding timestamp, label, and problematic features)
        exclude_columns = ['Row', 'Date', 'Time', 'Attack', 'Normal/Attack']  # Always exclude both label formats
        all_columns = [col for col in self.data.columns if col not in exclude_columns]
        
        # Remove problematic features from WADI topology remove_list
        remove_list = [
            '2B_AIT_002_PV', '2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS',
            'LEAK_DIFF_PRESSURE', 'PLANT_START_STOP_LOG', 'TOTAL_CONS_REQUIRED_FLOW'
        ]
        
        print(f"Removing {len(remove_list)} problematic features from WADI processing:")
        for feature in remove_list:
            if feature in all_columns:
                all_columns.remove(feature)
                print(f"  - Removing: {feature}")
            else:
                print(f"  - Feature not found (already removed): {feature}")
        
        print(f"Processing {len(all_columns)} features after removal (was {len([col for col in self.data.columns if col not in exclude_columns])})")
        
        # Store filtered columns
        self.columns = all_columns
        
        # Classify sensors vs actuators
        self.sensor_columns = []
        self.actuator_columns = []
        
        for col in self.columns:
            if any(keyword in col.upper() for keyword in ['STATUS', 'CO']):
                self.actuator_columns.append(col)
            else:
                self.sensor_columns.append(col)
        
        print(f"Classified {len(self.sensor_columns)} sensors and {len(self.actuator_columns)} actuators")
        
        # Check for any remaining problematic columns
        final_column_count = len(self.columns)
        print(f"Final feature count: {final_column_count}")
        
        # Ensure consistency across train/validation/test
        if hasattr(self, 'expected_feature_count'):
            if final_column_count != self.expected_feature_count:
                print(f"WARNING: Feature count mismatch! Expected {self.expected_feature_count}, got {final_column_count}")
                print(f"Current columns: {self.columns}")
        else:
            # Store expected count for future datasets
            self.expected_feature_count = final_column_count

        # Handle NaNs globally by filling with the mean of the column
        for col in self.columns:
            if self.data[col].isnull().any():
                # Ensure column is numeric before calculating mean
                numeric_col = pd.to_numeric(self.data[col], errors='coerce')
                col_mean = numeric_col.mean()
                self.data[col] = self.data[col].fillna(col_mean)
                print(f"  Warning: NaNs found in column {col}. Replaced with mean ({col_mean:.4f}).")

    def compute_initial_x0(self):
        """Compute initial 0-cell features based on normalization method."""
        print(f"Computing 0-cell features with normalization method: {self.normalization_method}")
        if self.use_first_order_differences:
            print("Including first-order differences (lag-1) in features")
            
        if self.normalization_method == "standard":
            return self._compute_standard_x0()
        elif self.normalization_method == "z_normalization":
            return self._compute_z_normalized_x0()
        elif self.normalization_method == "mixed_normalization":
            return self._compute_mixed_normalized_x0()
        elif self.normalization_method == "median_subtraction":
            return self._compute_median_subtraction_x0()
        elif self.normalization_method == "minmax":
            return self._compute_minmax_x0()
        elif self.normalization_method == "minmax_proper":
            return self._compute_minmax_proper_x0()
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")

    def _compute_standard_x0(self):
        """
        Original implementation: 3D feature vectors with one-hot encoding for actuators.
        """
        num_samples = len(self.data)
        num_components = len(self.columns)

        # Initialize tensor for 0-cell features
        features = torch.zeros((num_samples, num_components, self.feature_dim_0))

        print("Preprocessing features using standard method...")
        
        # Track which indices are actuators for later
        actuator_indices = []
        
        for i, col in enumerate(self.columns):
            values = self.data[col].values

            if is_actuator("WADI", col):
                # For actuators: one-hot encode, but state 0 is neutral [0,0,0]
                actuator_indices.append(i)
                
                for sample_idx in range(num_samples):
                    state = int(values[sample_idx])
                    # One-hot encode with max state = 2 (0, 1, 2)
                    # State 0 remains [0,0,0] (default initialized value)
                    if state == 1:
                        features[sample_idx, i, 1] = 1.0
                    elif state == 2:
                        features[sample_idx, i, 2] = 1.0
            else:
                # For sensors: store value in first dimension
                features[:, i, 0] = torch.tensor(values, dtype=torch.float)

        # For sensors, standardize values (after storing them)
        sensor_indices = [i for i, col in enumerate(self.columns) if not is_actuator("WADI", col)]
        
        # Calculate mean and std for each sensor (across all time points)
        sensor_means = torch.zeros(len(sensor_indices))
        sensor_stds = torch.zeros(len(sensor_indices))
        
        for idx, i in enumerate(sensor_indices):
            values = features[:, i, 0]
            sensor_means[idx] = values.mean()
            sensor_stds[idx] = values.std() + 1e-6  # Add small epsilon to avoid division by zero
            
            # Standardize: (x - mean) / std
            features[:, i, 0] = (values - sensor_means[idx]) / sensor_stds[idx]
        
        print(f"Normalized sensor features, mean: {sensor_means.mean():.4f}, std: {sensor_stds.mean():.4f}")
        print(f"Computed 0-cell features shape: {features.shape}")
        
        return features

    def _compute_mixed_normalized_x0(self):
        """
        Mixed normalization: z-score for sensors, min-max for actuators.
        Optionally includes first-order differences as additional features.
        """
        print("Preprocessing features using mixed normalization (z-norm for sensors, min-max for actuators)...")
        if self.use_first_order_differences:
            print("Computing first-order differences (current_value - previous_value)")
            
        num_samples = len(self.data)
        num_components = len(self.columns)
        
        # Create tensor with appropriate dimension (1D or 2D if differences enabled)
        features = torch.zeros(num_samples, num_components, self.feature_dim_0)
        
        for i, component in enumerate(self.columns):
            values = self.data[component].values
            
            if is_actuator("WADI", component):
                # Min-max normalization for actuators
                min_val = np.min(values)
                max_val = np.max(values)
                if max_val == min_val:
                    # Handle constant values
                    normalized_values = np.zeros_like(values)
                    print(f"  {component} (actuator): constant value={min_val:.4f}, kept as-is")
                else:
                    normalized_values = (values - min_val) / (max_val - min_val)
                    print(f"  {component} (actuator): min={min_val:.4f}, max={max_val:.4f}, range={max_val-min_val:.4f}")
            else:
                # Z-score normalization for sensors
                mean_val = np.mean(values)
                std_val = np.std(values) + 1e-6  # Avoid division by zero
                normalized_values = (values - mean_val) / std_val
                print(f"  {component} (sensor): mean={mean_val:.4f}, std={std_val:.4f}")
            
            # Store normalized values in the first dimension
            features[:, i, 0] = torch.tensor(normalized_values, dtype=torch.float32)
            
            # Compute first-order differences if enabled
            if self.use_first_order_differences:
                # Calculate lag-1 differences on NORMALIZED values (not raw values!)
                normalized_differences = np.zeros_like(normalized_values)
                normalized_differences[1:] = normalized_values[1:] - normalized_values[:-1]  # First sample difference = 0
                
                # Store normalized differences in the second dimension
                features[:, i, 1] = torch.tensor(normalized_differences, dtype=torch.float32)
                
                # Print statistics about normalized differences
                if i < 3:  # Only print for first few components to avoid spam
                    mean_abs_diff = np.mean(np.abs(normalized_differences[1:]))  # Skip first zero
                    max_abs_diff = np.max(np.abs(normalized_differences))
                    print(f"    Normalized first-order diff: mean_abs={mean_abs_diff:.6f}, max_abs={max_abs_diff:.6f}")
        
        print(f"Computed 0-cell features shape: {features.shape}")
        if self.use_first_order_differences:
            print(f"  Dimension 0: Normalized values")
            print(f"  Dimension 1: First-order differences (lag-1)")
        
        return features

    def _compute_z_normalized_x0(self):
        """
        New method: Z-normalization for both sensors and actuators, 1D features.
        """
        num_samples = len(self.data)
        num_components = len(self.columns)

        # Initialize tensor for 0-cell features (1D)
        features = torch.zeros((num_samples, num_components, 1))

        print("Preprocessing features using z-normalization for both sensors and actuators...")
        
        for i, col in enumerate(self.columns):
            values = self.data[col].values
            
            # Check for NaN or infinite values before processing
            if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                print(f"  WARNING: {col} contains NaN or infinite values before normalization!")
                # Replace NaN/inf with column mean
                finite_mask = np.isfinite(values)
                if np.any(finite_mask):
                    mean_val = np.mean(values[finite_mask])
                    values = np.where(finite_mask, values, mean_val)
                else:
                    values = np.zeros_like(values)
                print(f"  Fixed NaN/inf values in {col} using mean={mean_val:.4f}")
            
            # Apply z-normalization to all components
            mean_val = np.mean(values)
            std_val = np.std(values) + 1e-6  # Add small epsilon to avoid division by zero
            
            # Z-normalize: (x - mean) / std
            normalized_values = (values - mean_val) / std_val
            
            # Check for NaN/inf after normalization
            if np.any(np.isnan(normalized_values)) or np.any(np.isinf(normalized_values)):
                print(f"  ERROR: {col} z-normalization produced NaN/inf values!")
                print(f"  mean={mean_val:.4f}, std={std_val:.4f}")
                # Set to zero-mean values
                normalized_values = np.zeros_like(normalized_values)
                print(f"  Fixed by setting all values to 0.0")
            
            features[:, i, 0] = torch.tensor(normalized_values, dtype=torch.float)
            
            # Compute first-order differences if enabled
            if self.use_first_order_differences:
                # Calculate lag-1 differences on NORMALIZED values (not raw values!)
                normalized_differences = np.zeros_like(normalized_values)
                normalized_differences[1:] = normalized_values[1:] - normalized_values[:-1]  # First sample difference = 0
                
                # Store normalized differences in the second dimension
                features[:, i, 1] = torch.tensor(normalized_differences, dtype=torch.float32)
                
                # Print statistics about normalized differences
                if i < 3:  # Only print for first few components to avoid spam
                    mean_abs_diff = np.mean(np.abs(normalized_differences[1:]))  # Skip first zero
                    max_abs_diff = np.max(np.abs(normalized_differences))
                    print(f"    Normalized first-order diff: mean_abs={mean_abs_diff:.6f}, max_abs={max_abs_diff:.6f}")
            
            if i < 5:  # Only print first 5 for brevity
                print(f"  {col}: mean={mean_val:.4f}, std={std_val:.4f}")
            elif i == 5:
                print(f"  ... (suppressing further normalization details)")
        
        # Final check for NaN/inf in the entire feature tensor
        if torch.any(torch.isnan(features)) or torch.any(torch.isinf(features)):
            print("ERROR: Final feature tensor contains NaN or infinite values!")
            # Replace NaN/inf with zeros
            features = torch.where(torch.isfinite(features), features, torch.zeros_like(features))
            print("Fixed by replacing NaN/inf with zeros")
        
        print(f"Computed 0-cell features shape: {features.shape}")
        print(f"Feature tensor stats: min={features.min().item():.4f}, max={features.max().item():.4f}, mean={features.mean().item():.4f}")
        return features

    def _compute_median_subtraction_x0(self):
        """
        New method: Z-normalization for sensors, median subtraction for actuators, 1D features.
        """
        num_samples = len(self.data)
        num_components = len(self.columns)

        # Initialize tensor for 0-cell features (1D)
        features = torch.zeros((num_samples, num_components, 1))

        print("Preprocessing features using median subtraction (z-norm for sensors, median subtraction for actuators)...")
        
        for i, col in enumerate(self.columns):
            values = self.data[col].values
            
            # Check for NaN or infinite values before processing
            if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                print(f"  WARNING: {col} contains NaN or infinite values before normalization!")
                # Replace NaN/inf with column mean
                finite_mask = np.isfinite(values)
                if np.any(finite_mask):
                    mean_val = np.mean(values[finite_mask])
                    values = np.where(finite_mask, values, mean_val)
                else:
                    values = np.zeros_like(values)
                print(f"  Fixed NaN/inf values in {col} using mean={mean_val:.4f}")

            if is_actuator("WADI", col):
                # For actuators: just subtract median (no division)
                median_val = np.median(values)
                
                # Median subtraction: (x - median)
                normalized_values = values - median_val
                
                # Check for NaN/inf after normalization
                if np.any(np.isnan(normalized_values)) or np.any(np.isinf(normalized_values)):
                    print(f"  ERROR: {col} median subtraction produced NaN/inf values!")
                    print(f"  median={median_val:.4f}")
                    # Set to zero values
                    normalized_values = np.zeros_like(normalized_values)
                    print(f"  Fixed by setting all values to 0.0")
                
                features[:, i, 0] = torch.tensor(normalized_values, dtype=torch.float)
                
                print(f"  {col} (actuator): median={median_val:.4f}, range=[{np.min(normalized_values):.4f}, {np.max(normalized_values):.4f}]")
            else:
                # For sensors: z-normalization
                mean_val = np.mean(values)
                std_val = np.std(values) + 1e-6  # Add small epsilon to avoid division by zero
                
                # Z-normalize: (x - mean) / std
                normalized_values = (values - mean_val) / std_val
                
                # Check for NaN/inf after normalization
                if np.any(np.isnan(normalized_values)) or np.any(np.isinf(normalized_values)):
                    print(f"  ERROR: {col} z-normalization produced NaN/inf values!")
                    print(f"  mean={mean_val:.4f}, std={std_val:.4f}")
                    # Set to zero-mean values
                    normalized_values = np.zeros_like(normalized_values)
                    print(f"  Fixed by setting all values to 0.0")
                
                features[:, i, 0] = torch.tensor(normalized_values, dtype=torch.float)
                
                print(f"  {col} (sensor): mean={mean_val:.4f}, std={std_val:.4f}")
        
        # Final check for NaN/inf in the entire feature tensor
        if torch.any(torch.isnan(features)) or torch.any(torch.isinf(features)):
            print("ERROR: Final feature tensor contains NaN or infinite values!")
            # Replace NaN/inf with zeros
            features = torch.where(torch.isfinite(features), features, torch.zeros_like(features))
            print("Fixed by replacing NaN/inf with zeros")
        
        print(f"Computed 0-cell features shape: {features.shape}")
        print(f"Feature tensor stats: min={features.min().item():.4f}, max={features.max().item():.4f}, mean={features.mean().item():.4f}")
        return features

    def _compute_minmax_x0(self):
        """
        MinMax normalization for all components (sensors and actuators).
        Optionally includes first-order differences as additional features.
        """
        print("Preprocessing features using MinMax normalization (0-1 range) for all components...")
        if self.use_first_order_differences:
            print("Computing first-order differences (current_value - previous_value)")
            
        num_samples = len(self.data)
        num_components = len(self.columns)
        
        # Create tensor with appropriate dimension (1D or 2D if differences enabled)
        features = torch.zeros(num_samples, num_components, self.feature_dim_0)
        
        for i, component in enumerate(self.columns):
            values = self.data[component].values
            
            # Check for NaN or infinite values before processing
            if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                print(f"  WARNING: {component} contains NaN or infinite values before normalization!")
                # Replace NaN/inf with column mean
                finite_mask = np.isfinite(values)
                if np.any(finite_mask):
                    mean_val = np.mean(values[finite_mask])
                    values = np.where(finite_mask, values, mean_val)
                else:
                    values = np.zeros_like(values)
                print(f"  Fixed NaN/inf values in {component} using mean={mean_val:.4f}")
            
            # MinMax normalization for all components
            min_val = np.min(values)
            max_val = np.max(values)
            range_val = max_val - min_val
            
            # Handle case where min == max (constant values)
            if range_val == 0:
                # If all values are the same, set to 0.5 (middle of [0,1] range)
                normalized_values = np.full_like(values, 0.5, dtype=float)
                print(f"  {component}: constant value={min_val:.4f}, set to 0.5")
            else:
                # MinMax normalize: (x - min) / (max - min)
                normalized_values = (values - min_val) / range_val
                if i < 5:  # Only print details for first few components
                    print(f"  {component}: min={min_val:.4f}, max={max_val:.4f}, range={range_val:.4f}")
            
            # Store normalized values in first dimension
            features[:, i, 0] = torch.tensor(normalized_values, dtype=torch.float)
            
            # Compute first-order differences if enabled
            if self.use_first_order_differences:
                # Calculate lag-1 differences on NORMALIZED values (not raw values!)
                normalized_differences = np.zeros_like(normalized_values)
                normalized_differences[1:] = normalized_values[1:] - normalized_values[:-1]  # First sample difference = 0
                
                # Store normalized differences in the second dimension
                features[:, i, 1] = torch.tensor(normalized_differences, dtype=torch.float32)
                
                # Print statistics about normalized differences
                if i < 3:  # Only print for first few components to avoid spam
                    mean_abs_diff = np.mean(np.abs(normalized_differences[1:]))  # Skip first zero
                    max_abs_diff = np.max(np.abs(normalized_differences))
                    print(f"    MinMax first-order diff: mean_abs={mean_abs_diff:.6f}, max_abs={max_abs_diff:.6f}")
        
        # Final check for NaN/inf in the entire feature tensor
        if torch.any(torch.isnan(features)) or torch.any(torch.isinf(features)):
            print("ERROR: Final feature tensor contains NaN or infinite values!")
            # Replace NaN/inf with zeros
            features = torch.where(torch.isfinite(features), features, torch.zeros_like(features))
            print("Fixed by replacing NaN/inf with zeros")
        
        print(f"Computed 0-cell features shape: {features.shape}")
        print(f"Feature tensor stats: min={features.min().item():.4f}, max={features.max().item():.4f}, mean={features.mean().item():.4f}")
        if self.use_first_order_differences:
            print(f"  Dimension 0: MinMax normalized values (0-1 range)")
            print(f"  Dimension 1: First-order differences (lag-1)")
        
        return features

    def _compute_minmax_proper_x0(self):
        """
        Proper MinMax normalization using training data parameters (MinMaxScaler approach).
        This method implements the standard ML practice of fitting scaler on training data only.
        Optionally includes first-order differences as additional features.
        """
        print("Preprocessing features using Proper MinMax normalization (MinMaxScaler approach)...")
        if self.use_first_order_differences:
            print("Computing first-order differences (current_value - previous_value)")
            
        num_samples = len(self.data)
        num_components = len(self.columns)
        
        # Create tensor with appropriate dimension (1D or 2D if differences enabled)
        features = torch.zeros(num_samples, num_components, self.feature_dim_0)
        
        for i, component in enumerate(self.columns):
            values = self.data[component].values
            
            # Check for NaN or infinite values before processing
            if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                print(f"  WARNING: {component} contains NaN or infinite values before normalization!")
                # Replace NaN/inf with column mean
                finite_mask = np.isfinite(values)
                if np.any(finite_mask):
                    mean_val = np.mean(values[finite_mask])
                    values = np.where(finite_mask, values, mean_val)
                else:
                    values = np.zeros_like(values)
                print(f"  Fixed NaN/inf values in {component} using mean={mean_val:.4f}")
            
            # Use training data parameters for MinMax normalization (proper ML approach)
            if hasattr(self, 'train_min_vals') and hasattr(self, 'train_max_vals') and len(self.train_min_vals) > i:
                # Use pre-computed training parameters (validation/test data)
                min_val = self.train_min_vals[i]
                max_val = self.train_max_vals[i]
                range_val = max_val - min_val
                
                if range_val == 0:
                    normalized_values = np.full_like(values, 0.5, dtype=float)
                    print(f"  {component}: using training params, constant value={min_val:.4f}, set to 0.5")
                else:
                    normalized_values = (values - min_val) / range_val
                    if i < 5:  # Only print details for first few components
                        print(f"  {component}: using training params, min={min_val:.4f}, max={max_val:.4f}")
            else:
                # This is training data - compute and store parameters
                min_val = np.min(values)
                max_val = np.max(values)
                range_val = max_val - min_val
                
                # Store for future use (validation/test)
                if not hasattr(self, 'train_min_vals'):
                    self.train_min_vals = []
                    self.train_max_vals = []
                self.train_min_vals.append(min_val)
                self.train_max_vals.append(max_val)
                
                if range_val == 0:
                    normalized_values = np.full_like(values, 0.5, dtype=float)
                    print(f"  {component}: training data, constant value={min_val:.4f}, set to 0.5")
                else:
                    normalized_values = (values - min_val) / range_val
                    if i < 5:  # Only print details for first few components
                        print(f"  {component}: training data, min={min_val:.4f}, max={max_val:.4f}, range={range_val:.4f}")
            
            # Store normalized values in first dimension
            features[:, i, 0] = torch.tensor(normalized_values, dtype=torch.float)
            
            # Compute first-order differences if enabled
            if self.use_first_order_differences:
                # Calculate lag-1 differences on NORMALIZED values (not raw values!)
                normalized_differences = np.zeros_like(normalized_values)
                normalized_differences[1:] = normalized_values[1:] - normalized_values[:-1]  # First sample difference = 0
                
                # Store normalized differences in the second dimension
                features[:, i, 1] = torch.tensor(normalized_differences, dtype=torch.float32)
                
                # Print statistics about normalized differences
                if i < 3:  # Only print for first few components to avoid spam
                    mean_abs_diff = np.mean(np.abs(normalized_differences[1:]))  # Skip first zero
                    max_abs_diff = np.max(np.abs(normalized_differences))
                    print(f"    Proper MinMax first-order diff: mean_abs={mean_abs_diff:.6f}, max_abs={max_abs_diff:.6f}")
        
        # Final check for NaN/inf in the entire feature tensor
        if torch.any(torch.isnan(features)) or torch.any(torch.isinf(features)):
            print("ERROR: Final feature tensor contains NaN or infinite values!")
            # Replace NaN/inf with zeros
            features = torch.where(torch.isfinite(features), features, torch.zeros_like(features))
            print("Fixed by replacing NaN/inf with zeros")
        
        print(f"Computed 0-cell features shape: {features.shape}")
        print(f"Feature tensor stats: min={features.min().item():.4f}, max={features.max().item():.4f}, mean={features.mean().item():.4f}")
        if self.use_first_order_differences:
            print(f"  Dimension 0: Proper MinMax normalized values (0-1 range)")
            print(f"  Dimension 1: First-order differences (lag-1)")
        
        return features

    def compute_initial_x1(self):
        """
        Initialize 1-cell features based on connected 0-cells.
        Concatenate features and optionally add GECO features.
        """
        if self.normalization_method == "standard":
            return self._compute_standard_x1()
        else:
            return self._compute_new_x1()

    def _compute_standard_x1(self):
        """
        Original implementation for 1-cell features.
        """
        if self.use_geco_features:
            print("Computing initial 1-cell features (concatenation + GECO method)...")
        else:
            print("Computing initial 1-cell features (concatenation method)...")
            
        num_samples = len(self.data)

        # Get row and column dictionaries for 0-1 incidence matrix
        row_dict, col_dict, _ = self.complex.incidence_matrix(0, 1, index=True)
        
        # Use the column dictionary to get all 1-cells
        cells_1 = list(col_dict.keys())
        
        # Create a mapping from component name to index in our data
        component_to_idx = {comp: i for i, comp in enumerate(self.columns)}
        
        # Filter 1-cells to only include those where both boundary nodes exist in our data
        valid_cells_1 = []
        valid_cell_indices = []
        
        for cell_idx, cell in enumerate(cells_1):
            boundary_nodes = list(cell)
            if len(boundary_nodes) == 2:
                # Check if both boundary nodes exist in our dataset
                if all(node in component_to_idx for node in boundary_nodes):
                    valid_cells_1.append(cell)
                    valid_cell_indices.append(cell_idx)
        
        num_1_cells = len(valid_cells_1)
        print(f"Found {len(cells_1)} total 1-cells, {num_1_cells} valid for current dataset")

        # Create tensor for 1-cell features
        x_1 = torch.zeros((num_samples, num_1_cells, self.feature_dim_1))

        # For each valid 1-cell (edge), concatenate the features of its boundary 0-cells
        for new_cell_idx, cell in enumerate(valid_cells_1):
            boundary_nodes = list(cell)

            # Get indices of boundary nodes in our dataset
            node_indices = [component_to_idx[node] for node in boundary_nodes]

            # Get GECO features if enabled
            geco_features = None
            if self.use_geco_features:
                geco_features = self.wadi_complex.get_geco_edge_features(boundary_nodes)

            # Concatenate the features of the boundary nodes
            for sample_idx in range(num_samples):
                # Concatenate features of connected 0-cells (first 6 dimensions)
                node1_feat = self.x_0[sample_idx, node_indices[0]]  # First 3 dimensions
                node2_feat = self.x_0[sample_idx, node_indices[1]]  # Next 3 dimensions
                edge_feat = torch.cat([node1_feat, node2_feat], dim=0)  # 6D concatenated feature
                
                # Add GECO features if enabled
                if self.use_geco_features:
                    if geco_features is not None:
                        # Simplified GECO features: only normalized strength
                        geco_feat = torch.tensor([
                            geco_features['geco_normalized_strength']  # Only normalized strength
                        ], dtype=torch.float32)
                    else:
                        # Default GECO features for non-GECO edges
                        geco_feat = torch.tensor([0.0], dtype=torch.float32)  # Zero strength for non-GECO
                    
                    # Concatenate original features with simplified GECO features
                    edge_feat = torch.cat([edge_feat, geco_feat], dim=0)  # 3D total feature (2D + 1D GECO)
                
                x_1[sample_idx, new_cell_idx] = edge_feat

        print(f"Computed 1-cell features shape: {x_1.shape}")
        return x_1

    def _compute_new_x1(self):
        """
        New implementation for 1-cell features with 1D or 2D 0-cells.
        Handles concatenation of 0-cell features regardless of their dimensionality.
        Optionally adds first-order differences for edges if enabled.
        """
        if self.use_geco_features:
            print("Computing initial 1-cell features (concatenation + GECO method)...")
        else:
            print("Computing initial 1-cell features (concatenation method)...")
            
        if not self.use_first_order_differences_edges:
            print("  First-order differences DISABLED for 1-cells (edges)")
            
        num_samples = len(self.data)

        # Get row and column dictionaries for 0-1 incidence matrix
        row_dict, col_dict, _ = self.complex.incidence_matrix(0, 1, index=True)
        
        # Use the column dictionary to get all 1-cells
        cells_1 = list(col_dict.keys())
        
        # Create a mapping from component name to index in our data
        component_to_idx = {comp: i for i, comp in enumerate(self.columns)}
        
        # Filter 1-cells to only include those where both boundary nodes exist in our data
        valid_cells_1 = []
        valid_cell_indices = []
        
        for cell_idx, cell in enumerate(cells_1):
            boundary_nodes = list(cell)
            if len(boundary_nodes) == 2:
                # Check if both boundary nodes exist in our dataset
                if all(node in component_to_idx for node in boundary_nodes):
                    valid_cells_1.append(cell)
                    valid_cell_indices.append(cell_idx)
        
        num_1_cells = len(valid_cells_1)
        print(f"Found {len(cells_1)} total 1-cells, {num_1_cells} valid for current dataset")

        # Create tensor for 1-cell features
        x_1 = torch.zeros((num_samples, num_1_cells, self.feature_dim_1))

        # Store edge features for first-order differences computation if needed
        if self.use_first_order_differences_edges:
            edge_base_features = torch.zeros((num_samples, num_1_cells, self.feature_dim_1 - 1))

        # For each valid 1-cell (edge), concatenate the features of its boundary 0-cells
        for new_cell_idx, cell in enumerate(valid_cells_1):
            boundary_nodes = list(cell)

            # Get indices of boundary nodes in our dataset
            node_indices = [component_to_idx[node] for node in boundary_nodes]

            # Get GECO features if enabled
            geco_features = None
            if self.use_geco_features:
                geco_features = self.wadi_complex.get_geco_edge_features(boundary_nodes)

            # Concatenate the features of the boundary nodes
            for sample_idx in range(num_samples):
                # Concatenate features of connected 0-cells
                # Handle both 1D and 2D 0-cell features automatically
                node1_feat = self.x_0[sample_idx, node_indices[0]]  # Could be 1D or 2D
                node2_feat = self.x_0[sample_idx, node_indices[1]]  # Could be 1D or 2D
                
                # For edge features, only use the FIRST dimension (original value, not first-order differences)
                if self.use_first_order_differences:
                    # Extract only the normalized values (dimension 0), not the differences (dimension 1)
                    node1_value = node1_feat[0:1]  # Keep as 1D tensor
                    node2_value = node2_feat[0:1]  # Keep as 1D tensor
                else:
                    # Use the full feature (should be 1D anyway)
                    node1_value = node1_feat
                    node2_value = node2_feat
                
                edge_feat = torch.cat([node1_value, node2_value], dim=0)  # Concatenated feature
                
                # Add GECO features if enabled
                if self.use_geco_features:
                    if geco_features is not None:
                        # Simplified GECO features: only normalized strength
                        geco_feat = torch.tensor([
                            geco_features['geco_normalized_strength']  # Only normalized strength
                        ], dtype=torch.float32)
                    else:
                        # Default GECO features for non-GECO edges
                        geco_feat = torch.tensor([0.0], dtype=torch.float32)  # Zero strength for non-GECO
                    
                    # Concatenate original features with simplified GECO features
                    edge_feat = torch.cat([edge_feat, geco_feat], dim=0)  # Total feature with GECO
                
                # Store base features if we need to compute first-order differences
                if self.use_first_order_differences_edges:
                    edge_base_features[sample_idx, new_cell_idx] = edge_feat
                    
                    # Compute first-order differences for edges
                    if sample_idx == 0:
                        # First sample has no previous sample, so difference = 0
                        edge_diff = torch.zeros(1, dtype=torch.float32)
                    else:
                        # Compute difference from previous sample
                        prev_feat = edge_base_features[sample_idx - 1, new_cell_idx]
                        edge_diff = torch.mean(edge_feat - prev_feat).unsqueeze(0)  # Average difference as 1D
                    
                    # Concatenate base features with difference
                    final_edge_feat = torch.cat([edge_feat, edge_diff], dim=0)
                else:
                    # No differences for edges
                    final_edge_feat = edge_feat
                
                x_1[sample_idx, new_cell_idx] = final_edge_feat

        # Add cosine similarity features for edges (like SWAT implementation)
        if hasattr(self.wadi_complex, 'get_embedding_similarity'):
            print("Adding cosine similarity features for 1-cells...")
            
            # Create new tensor with additional dimension for cosine similarity
            x_1_with_similarity = torch.zeros((num_samples, num_1_cells, self.feature_dim_1 + 1))
            
            for new_cell_idx, cell in enumerate(valid_cells_1):
                boundary_nodes = list(cell)
                
                # Get cosine similarity between the two nodes
                try:
                    similarity = self.wadi_complex.get_embedding_similarity(boundary_nodes[0], boundary_nodes[1])
                    similarity_tensor = torch.tensor(similarity, dtype=torch.float32)
                except Exception as e:
                    print(f"Warning: Could not compute similarity for edge {boundary_nodes}: {e}")
                    similarity_tensor = torch.tensor(0.0, dtype=torch.float32)
                
                # Add similarity to all samples for this edge
                for sample_idx in range(num_samples):
                    # Copy original features
                    x_1_with_similarity[sample_idx, new_cell_idx, :-1] = x_1[sample_idx, new_cell_idx]
                    # Add similarity as the last feature
                    x_1_with_similarity[sample_idx, new_cell_idx, -1] = similarity_tensor
            
            # Update x_1 and feature dimension
            x_1 = x_1_with_similarity
            self.feature_dim_1 += 1
            print(f"Added cosine similarity features. New 1-cell feature dimension: {self.feature_dim_1}D")

        print(f"Computed 1-cell features shape: {x_1.shape}")
        if self.use_first_order_differences:
            print(f"  Each 1-cell uses only VALUES (not differences) from {self.feature_dim_0}D 0-cell features" + 
                  (f" + 1D GECO" if self.use_geco_features else ""))
        if self.use_first_order_differences_edges:
            print(f"  + 1D first-order differences for edges")
        if hasattr(self.wadi_complex, 'get_embedding_similarity'):
            print(f"  + 1D cosine similarity features")
        
        return x_1

    def compute_initial_x2(self):
        """
        Initialize 2-cell features based on configuration.
        """
        if self.normalization_method == "standard":
            return self._compute_standard_x2()
        else:
            return self._compute_new_x2()

    def _compute_standard_x2(self):
        """
        Original implementation for 2-cell features.
        """
        print("Computing initial 2-cell features...")
        num_samples = len(self.data)
        
        # Get 2-cells (subsystems) using the correct TopoNetX method
        _, col_dict_12, _ = self.complex.incidence_matrix(1, 2, index=True)
        
        # Use the column dictionary to get all 2-cells
        cells_2 = list(col_dict_12.keys())
        num_2_cells = len(cells_2)
        
        if num_2_cells == 0:
            print("No 2-cells found, creating dummy 2-cell features")
            return torch.zeros((num_samples, 1, self.feature_dim_2))
        
        print(f"Found {num_2_cells} 2-cells (subsystems)")
        
        # Create tensor for 2-cell features
        x_2 = torch.zeros((num_samples, num_2_cells, self.feature_dim_2))
        
        # Get component to index mapping
        component_to_idx = {comp: i for i, comp in enumerate(self.columns)}
        
        # For each 2-cell, compute average features of its components
        for cell_idx, cell in enumerate(cells_2):
            # Each 2-cell contains multiple 0-cells
            # Extract 0-cells directly from the frozenset (they're already listed there)
            try:
                # Extract 0-cells directly from the 2-cell frozenset
                node_set = set(cell)
                
                # Get indices of these 0-cells
                node_indices = []
                for node in node_set:
                    if node in component_to_idx:
                        node_indices.append(component_to_idx[node])
                
                # If we have valid nodes, average their features
                if node_indices:
                    for sample_idx in range(num_samples):
                        # Average features of all 0-cells in the group
                        face_feat = torch.zeros(self.feature_dim_2)
                        for node_idx in node_indices:
                            face_feat += self.x_0[sample_idx, node_idx]
                        face_feat /= len(node_indices)
                        x_2[sample_idx, cell_idx] = face_feat
                else:
                    print(f"Warning: No valid 0-cells found for 2-cell {cell}")

            except Exception as e:
                print(f"Warning: Error computing 2-cell features for cell {cell}: {e}")
                print(f"Cell contents: {cell}")
        
        print(f"Computed 2-cell features shape: {x_2.shape}")
        return x_2

    def _compute_new_x2(self):
        """
        New implementation for 2-cell features with enhanced statistics.
        """
        if self.use_enhanced_2cell_features:
            return self._compute_enhanced_x2()
        else:
            return self._compute_simple_x2()

    def _compute_simple_x2(self):
        """
        Simple 2-cell features: average of 1D 0-cell features.
        """
        print("Computing simple 2-cell features (average of 1D features)...")
        num_samples = len(self.data)
        
        # Get 2-cells (subsystems) using the correct TopoNetX method
        _, col_dict_12, _ = self.complex.incidence_matrix(1, 2, index=True)
        
        # Use the column dictionary to get all 2-cells
        cells_2 = list(col_dict_12.keys())
        num_2_cells = len(cells_2)
        
        if num_2_cells == 0:
            print("No 2-cells found, creating dummy 2-cell features")
            return torch.zeros((num_samples, 1, self.feature_dim_2))
        
        print(f"Found {num_2_cells} 2-cells (subsystems)")
        
        # Create tensor for 2-cell features
        x_2 = torch.zeros((num_samples, num_2_cells, self.feature_dim_2))
        
        # Get component to index mapping
        component_to_idx = {comp: i for i, comp in enumerate(self.columns)}
        
        # For each 2-cell, compute average features of its components
        for cell_idx, cell in enumerate(cells_2):
            try:
                # Extract 0-cells directly from the 2-cell frozenset
                node_set = set(cell)
                
                # Get indices of these 0-cells
                node_indices = []
                for node in node_set:
                    if node in component_to_idx:
                        node_indices.append(component_to_idx[node])
                
                # If we have valid nodes, average their features
                if node_indices:
                    for sample_idx in range(num_samples):
                        # Average features of all 0-cells in the group
                        face_feat = torch.zeros(self.feature_dim_2)
                        for node_idx in node_indices:
                            face_feat += self.x_0[sample_idx, node_idx]
                        face_feat /= len(node_indices)
                        x_2[sample_idx, cell_idx] = face_feat
                else:
                    print(f"Warning: No valid 0-cells found for 2-cell {cell}")

            except Exception as e:
                print(f"Warning: Error computing 2-cell features for cell {cell}: {e}")
                print(f"Cell contents: {cell}")
        
        print(f"Computed 2-cell features shape: {x_2.shape}")
        return x_2

    def _compute_enhanced_x2(self):
        """
        Compute enhanced 2-cell features with 4D vectors:
        [sensor_mean, sensor_std, actuator_median, actuator_range]
        """
        print("Computing enhanced 2-cell features (4D: mean/std for sensors, median/range for actuators)")
            
        num_samples = len(self.data)
        
        _, col_dict_12, _ = self.complex.incidence_matrix(1, 2, index=True)
        cells_2 = list(col_dict_12.keys())
        num_2_cells = len(cells_2)
        
        if num_2_cells == 0:
            return torch.zeros((num_samples, 1, self.feature_dim_2))
        
        x_2 = torch.zeros((num_samples, num_2_cells, self.feature_dim_2))
        
        cell_to_nodes = {}
        for cell_idx, cell in enumerate(cells_2):
            node_set = set(cell)
            node_indices = [self.component_to_idx.get(node) for node in node_set]
            node_indices = [idx for idx in node_indices if idx is not None]
            
            node_names = [self.columns[i] for i in node_indices]
            
            sensor_indices = [i for i, name in zip(node_indices, node_names) if not is_actuator("WADI", name)]
            actuator_indices = [i for i, name in zip(node_indices, node_names) if is_actuator("WADI", name)]
            cell_to_nodes[cell_idx] = (sensor_indices, actuator_indices)

        for sample_idx in range(num_samples):
            sample_x0 = self.x_0[sample_idx, :, 0]  # Use only the first dimension (normalized values)
            
            for cell_idx in range(num_2_cells):
                sensor_indices, actuator_indices = cell_to_nodes[cell_idx]
                
                enhanced_feat = torch.zeros(self.feature_dim_2)
                
                # Original 4D enhanced features: sensor mean/std, actuator median/range
                if sensor_indices:
                    sensor_features = sample_x0[sensor_indices]
                    enhanced_feat[0] = torch.mean(sensor_features)
                    enhanced_feat[1] = torch.std(sensor_features) if len(sensor_indices) > 1 else 0.0
                
                if actuator_indices:
                    actuator_features = sample_x0[actuator_indices]
                    enhanced_feat[2] = torch.median(actuator_features)
                    enhanced_feat[3] = torch.max(actuator_features) - torch.min(actuator_features)  # range
                
                x_2[sample_idx, cell_idx] = enhanced_feat
        
        print(f"Computed enhanced 2-cell features shape: {x_2.shape}")
        return x_2

    def create_adjacency_matrices(self):
        """Create adjacency matrices for the complex."""
        print("Creating adjacency matrices...")
        
        # Get topology matrices
        self.a0, self.a1, self.coa2, self.b1, self.b2 = self.get_neighborhood_matrix()
        
        print("Adjacency matrices created successfully")

    def get_neighborhood_matrix(self):
        """
        Compute the neighborhood matrices for the WADI complex.

        Returns
        -------
        tuple of torch.sparse.Tensor
            Adjacency, coadjacency and incidence matrices
        """
        print("Computing WADI neighborhood matrices...")

        # Get adjacency matrices
        a0 = torch.from_numpy(self.complex.adjacency_matrix(0, 1).todense()).to_sparse()
        a1 = torch.from_numpy(self.complex.adjacency_matrix(1, 2).todense()).to_sparse()

        # Get incidence matrices
        b1 = torch.from_numpy(self.complex.incidence_matrix(0, 1).todense()).to_sparse()
        b2 = torch.from_numpy(self.complex.incidence_matrix(1, 2).todense()).to_sparse()

        # Compute coadjacency matrix for 2-cells
        B = self.complex.incidence_matrix(1, 2)
        A = B.T @ B
        A.setdiag(0)
        coa2 = torch.from_numpy(A.todense()).to_sparse()

        print(f"WADI matrix dimensions - a0: {a0.shape}, a1: {a1.shape}, coa2: {coa2.shape}, b1: {b1.shape}, b2: {b2.shape}")
        return a0, a1, coa2, b1, b2

    def create_temporal_samples(self):
        """
        Subsample the data for temporal mode.
        """
        if self.temporal_sample_rate > 1:
            sampled_indices = list(range(0, len(self.data), self.temporal_sample_rate))
            self.data = self.data.iloc[sampled_indices].reset_index(drop=True)
            self.labels = self.labels[sampled_indices]
            print(f"Applied temporal sampling every {self.temporal_sample_rate} timesteps: {len(self.data)} samples remaining")

        if self.temporal_mode:
            self.effective_length = len(self.data) - self.n_input
            print(f"Using temporal mode with sequence length {self.n_input}, effective samples: {self.effective_length}")
        else:
            self.effective_length = len(self.data)

        print(f"Initialized WADIDataset with {self.effective_length} samples")
        print(f"Feature dimensions - 0-cells: {self.x_0.shape} (dim={self.feature_dim_0}), 1-cells: {self.x_1.shape} (dim={self.feature_dim_1}), 2-cells: {self.x_2.shape} (dim={self.feature_dim_2})")
        
        # Debug: Print sample features
        self.print_sample_features()

    def print_sample_features(self):
        """Print sample features for debugging."""
        if len(self.data) == 0:
            return
            
        print("\n=== DEBUG: WADI Sample Features ===")
        
        # Print one 0-cell feature
        sample_0_cell = self.x_0[0, 0]  # First sample, first 0-cell
        print(f"Sample WADI 0-cell feature (component '{self.columns[0]}'): {sample_0_cell.numpy()}")
        
        # Print one 1-cell feature
        if self.x_1.shape[1] > 0:
            sample_1_cell = self.x_1[0, 0]  # First sample, first 1-cell
            print(f"Sample WADI 1-cell feature: {sample_1_cell.numpy()}")
            if self.use_geco_features:
                print(f"  - First 6 dims (concatenated 0-cell features): {sample_1_cell[:6].numpy()}")
                print(f"  - Last 2 dims (GECO features - strength, equation_type): {sample_1_cell[6:].numpy()}")
        
        # Print one 2-cell feature
        if self.x_2.shape[1] > 0:
            sample_2_cell = self.x_2[0, 0]  # First sample, first 2-cell
            print(f"Sample WADI 2-cell feature: {sample_2_cell.numpy()}")
            print(f"  - All 3 dims (averaged 0-cell features): {sample_2_cell.numpy()}")
        
        print("=== END WADI DEBUG ===\n")

    def __len__(self):
        """Return the number of samples."""
        return self.effective_length

    def __getitem__(self, idx):
        """Return a sample from the dataset."""
        if self.temporal_mode:
            # Return sequence of timesteps and next timestep as target
            if idx >= self.effective_length:
                idx = self.effective_length - 1  # Handle boundary case
                
            # Get sequence of inputs (t-n_input to t-1)
            seq_x0 = self.x_0[idx:idx+self.n_input]  # [n_input, n_nodes, feat_dim]
            seq_x1 = self.x_1[idx:idx+self.n_input]  # [n_input, n_edges, feat_dim]
            seq_x2 = self.x_2[idx:idx+self.n_input]  # [n_input, n_2cells, feat_dim]
            
            # Target is the next timestep (t)
            target_x0 = self.x_0[idx+self.n_input]  # [n_nodes, feat_dim]
            target_x1 = self.x_1[idx+self.n_input]  # [n_edges, feat_dim]
            target_x2 = self.x_2[idx+self.n_input]  # [n_2cells, feat_dim]
            
            # Label of the target timestep
            label = self.labels[idx+self.n_input]
            
            return {
                'seq_x0': seq_x0,
                'seq_x1': seq_x1,
                'seq_x2': seq_x2,
                'target_x0': target_x0,
                'target_x1': target_x1,
                'target_x2': target_x2,
                'a0': self.a0,
                'a1': self.a1,
                'coa2': self.coa2,
                'b1': self.b1,
                'b2': self.b2,
                'label': label
            }
        else:
            # Original code for single timestep
            return (   
                self.x_0[idx],
                self.x_1[idx],
                self.x_2[idx],
                self.a0,
                self.a1,
                self.coa2,
                self.b1,
                self.b2,
                self.labels[idx]
            ) 

    def get_num_components(self):
        """Return the number of components for loss normalization."""
        return len(self.columns)
    
    def get_feature_scale_info(self):
        """Return scaling information for debugging."""
        return {
            'num_components': len(self.columns),
            'num_0_cells': self.x_0.shape[1],
            'num_1_cells': self.x_1.shape[1],
            'num_2_cells': self.x_2.shape[1],
            'total_features': self.x_0.shape[1] + self.x_1.shape[1] + self.x_2.shape[1]
        } 