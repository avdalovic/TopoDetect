import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


from src.utils.attack_utils import is_actuator


class SWaTDataset(Dataset):
    """Dataset class for SWAT anomaly detection."""
    def __init__(self, data, swat_complex, feature_dim_0=3, n_input=10, temporal_mode=False, temporal_sample_rate=1, use_geco_features=False, 
                 normalization_method="standard", use_enhanced_2cell_features=False, seed=None):
        """
        Initialize the SWAT dataset for anomaly detection.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing the SWAT measurements and labels
        swat_complex : toponetx.CombinatorialComplex
            Combinatorial complex representing the SWAT system topology
        feature_dim_0 : int, default=3
            Dimension of feature vectors for 0-cells (3 to accommodate one-hot encoding)
        n_input : int, default=10
            Number of input timesteps for temporal mode
        temporal_mode : bool, default=False
            Whether to use temporal mode
        temporal_sample_rate : int, default=1
            Sampling rate for temporal mode
        use_geco_features : bool, default=False
            Whether to use enhanced edge features with GECO information
        normalization_method : str, default="standard"
            Normalization method: "standard" (current), "z_normalization", "mixed_normalization"
        use_enhanced_2cell_features : bool, default=False
            Whether to use enhanced 2-cell features (4D: mean_sensors, sd_sensors, median_actuators, iqr_actuators)
        seed : int, optional
            Random seed for reproducibility
        """
        self.data = data
        self.swat_complex = swat_complex  # Store the wrapper object
        self.complex = swat_complex.get_complex()  # Get the actual complex
        self.feature_dim_0 = feature_dim_0
        self.n_input = n_input
        self.temporal_mode = temporal_mode
        self.temporal_sample_rate = temporal_sample_rate
        self.use_geco_features = use_geco_features
        self.normalization_method = normalization_method
        self.use_enhanced_2cell_features = use_enhanced_2cell_features
        self.seed = seed
        
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        print(f"Using normalization method: {normalization_method}")
        print(f"Using enhanced 2-cell features: {use_enhanced_2cell_features}")
        if seed is not None:
            print(f"Using seed: {seed}")
        
        # Set feature dimensions based on configuration
        # Determine actual 0-cell feature dimension based on normalization method
        if normalization_method == "standard":
            actual_feature_dim_0 = 3  # Original 3D features
        else:
            actual_feature_dim_0 = 1  # New methods use 1D features
            
        if self.use_geco_features:
            if actual_feature_dim_0 == 3:  # Original implementation
                self.feature_dim_1 = actual_feature_dim_0 * 2 + 1  # 2d + 1D GECO = 7D
            else:  # New implementation with 1D 0-cells
                self.feature_dim_1 = 2 + 1  # 2D concatenated + 1D GECO = 3D
            print(f"Using simplified GECO edge features: {self.feature_dim_1}D")
        else:
            if actual_feature_dim_0 == 3:  # Original implementation
                self.feature_dim_1 = actual_feature_dim_0 * 2  # 6D concatenated features
            else:  # New implementation with 1D 0-cells
                self.feature_dim_1 = 2  # 2D concatenated features
            print(f"Using standard edge features: {self.feature_dim_1}D")
        
        # Override the feature_dim_0 for consistency
        self.feature_dim_0 = actual_feature_dim_0
        
        # Set 2-cell feature dimensions
        if use_enhanced_2cell_features:
            self.feature_dim_2 = 4  # 4D: mean/std for sensors, median/range for actuators
            print(f"Using simplified 2-cell features: {self.feature_dim_2}D")
        else:
            self.feature_dim_2 = 1  # 1D for new methods without enhancement
            print(f"Using standard 2-cell features: {self.feature_dim_2}D")
        
        # Process data and create features
        self.preprocess_features()
        self.node_index_map = self.swat_complex._get_node_index_map()
        self.x_0 = self.compute_initial_x0()
        self.x_1 = self.compute_initial_x1()
        self.x_2 = self.compute_initial_x2(self.node_index_map)
        self.labels = self.data['Normal/Attack'].values
        
        # Create adjacency matrices for the complex
        self.create_adjacency_matrices()
        
        # Handle temporal mode
        if temporal_mode:
            self.create_temporal_samples()
        else:
            self.effective_length = len(self.data)

    def preprocess_features(self):
        """
        Extract and preprocess features from the data.
        """
        # Extract features and labels
        self.labels = self.data['Normal/Attack'].map(lambda x: 0 if (x == 'False' or x == 0 or x == 0.0) else 1).values

        # Get sensor and actuator columns (excluding timestamp and label)
        self.columns = [col for col in self.data.columns if col not in ['Timestamp', 'Normal/Attack']]

        # Handle NaNs globally by filling with the mean of the column
        for col in self.columns:
            if self.data[col].isnull().any():
                # Ensure column is numeric before calculating mean
                numeric_col = pd.to_numeric(self.data[col], errors='coerce')
                col_mean = numeric_col.mean()
                self.data[col] = self.data[col].fillna(col_mean)
                print(f"  Warning: NaNs found in column {col}. Replaced with mean ({col_mean:.4f}).")

    def compute_initial_x0(self):
        """
        Preprocess raw features to create feature vectors based on normalization method.
        """
        if self.normalization_method == "standard":
            return self._compute_standard_x0()
        elif self.normalization_method == "z_normalization":
            return self._compute_z_normalized_x0()
        elif self.normalization_method == "mixed_normalization":
            return self._compute_mixed_normalized_x0()
        elif self.normalization_method == "median_subtraction":
            return self._compute_median_subtraction_x0()
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

            if is_actuator("SWAT", col):
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
        sensor_indices = [i for i, col in enumerate(self.columns) if not is_actuator("SWAT", col)]
        
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
        New method: Z-normalization for sensors, min-max normalization for actuators, 1D features.
        """
        num_samples = len(self.data)
        num_components = len(self.columns)

        # Initialize tensor for 0-cell features (1D)
        features = torch.zeros((num_samples, num_components, 1))

        print("Preprocessing features using mixed normalization (z-norm for sensors, min-max for actuators)...")
        
        for i, col in enumerate(self.columns):
            values = self.data[col].values

            if is_actuator("SWAT", col):
                # For actuators: min-max normalization to [0, 1] range
                min_val = np.min(values)
                max_val = np.max(values)
                range_val = max_val - min_val
                
                # Handle case where min == max (constant values)
                if range_val == 0:
                    # If all values are the same, keep the original constant value
                    normalized_values = np.full_like(values, max_val, dtype=float)
                    print(f"  {col} (actuator): constant value={min_val:.4f}, kept as-is")
                else:
                    # Min-max normalize: (x - min) / (max - min)
                    normalized_values = (values - min_val) / range_val
                    print(f"  {col} (actuator): min={min_val:.4f}, max={max_val:.4f}, range={range_val:.4f}")
                
                features[:, i, 0] = torch.tensor(normalized_values, dtype=torch.float)
                
            else:
                # For sensors: z-normalization
                mean_val = np.mean(values)
                std_val = np.std(values) + 1e-6  # Add small epsilon to avoid division by zero
                
                # Z-normalize: (x - mean) / std
                normalized_values = (values - mean_val) / std_val
                features[:, i, 0] = torch.tensor(normalized_values, dtype=torch.float)
                
                print(f"  {col} (sensor): mean={mean_val:.4f}, std={std_val:.4f}")
        
        print(f"Computed 0-cell features shape: {features.shape}")
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
            
            # Apply z-normalization to all components
            mean_val = np.mean(values)
            std_val = np.std(values) + 1e-6  # Add small epsilon to avoid division by zero
            
            # Z-normalize: (x - mean) / std
            normalized_values = (values - mean_val) / std_val
            features[:, i, 0] = torch.tensor(normalized_values, dtype=torch.float)
            
            if i < 5:  # Only print first 5 for brevity
                print(f"  {col}: mean={mean_val:.4f}, std={std_val:.4f}")
            elif i == 5:
                print(f"  ... (suppressing further normalization details)")
        
        print(f"Computed 0-cell features shape: {features.shape}")
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

            if is_actuator("SWAT", col):
                # For actuators: just subtract median (no division)
                median_val = np.median(values)
                
                # Median subtraction: (x - median)
                normalized_values = values - median_val
                features[:, i, 0] = torch.tensor(normalized_values, dtype=torch.float)
                
                print(f"  {col} (actuator): median={median_val:.4f}, range=[{np.min(normalized_values):.4f}, {np.max(normalized_values):.4f}]")
            else:
                # For sensors: z-normalization
                mean_val = np.mean(values)
                std_val = np.std(values) + 1e-6  # Add small epsilon to avoid division by zero
                
                # Z-normalize: (x - mean) / std
                normalized_values = (values - mean_val) / std_val
                features[:, i, 0] = torch.tensor(normalized_values, dtype=torch.float)
                
                print(f"  {col} (sensor): mean={mean_val:.4f}, std={std_val:.4f}")
        
        print(f"Computed 0-cell features shape: {features.shape}")
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
                geco_features = self.swat_complex.get_geco_edge_features(boundary_nodes)

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
        New implementation for 1-cell features with 1D 0-cells.
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
                geco_features = self.swat_complex.get_geco_edge_features(boundary_nodes)

            # Concatenate the features of the boundary nodes
            for sample_idx in range(num_samples):
                # Concatenate features of connected 0-cells (2D: 1D + 1D)
                node1_feat = self.x_0[sample_idx, node_indices[0]]  # 1D
                node2_feat = self.x_0[sample_idx, node_indices[1]]  # 1D
                edge_feat = torch.cat([node1_feat, node2_feat], dim=0)  # 2D concatenated feature
                
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

    def compute_initial_x2(self, node_index_map):
        """
        Initialize 2-cell features based on configuration.
        """
        if self.normalization_method == "standard":
            return self._compute_standard_x2(node_index_map)
        else:
            return self._compute_new_x2(node_index_map)

    def _compute_standard_x2(self, node_index_map=None):
        """
        Original implementation for 2-cell features.
        """
        print("Computing initial 2-cell features...")
        num_samples = len(self.data)
        
        # Get 2-cells (PLC zones) using the correct TopoNetX method
        row_dict_01, _, _ = self.complex.incidence_matrix(0, 1, index=True)
        _, col_dict_12, _ = self.complex.incidence_matrix(1, 2, index=True)
        
        # Use the column dictionary to get all 2-cells
        cells_2 = list(col_dict_12.keys())
        num_2_cells = len(cells_2)
        
        if num_2_cells == 0:
            print("No 2-cells found, creating dummy 2-cell features")
            return torch.zeros((num_samples, 1, self.feature_dim_2))
        
        print(f"Found {num_2_cells} 2-cells (PLC zones)")
        
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
                
                # Get indices of these 0-cells in the row dictionary
                node_indices = []
                for node in node_set:
                    node_frozenset = frozenset({node})
                    if node_frozenset in row_dict_01:
                        node_indices.append(row_dict_01[node_frozenset])
                
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

    def _compute_new_x2(self, node_index_map):
        """
        New implementation for 2-cell features with enhanced statistics.
        """
        if self.use_enhanced_2cell_features:
            return self._compute_enhanced_x2(node_index_map)
        else:
            return self._compute_simple_x2(node_index_map)

    def _compute_simple_x2(self, node_index_map=None):
        """
        Simple 2-cell features: average of 1D 0-cell features.
        """
        print("Computing simple 2-cell features (average of 1D features)...")
        num_samples = len(self.data)
        
        # Get 2-cells (PLC zones) using the correct TopoNetX method
        row_dict_01, _, _ = self.complex.incidence_matrix(0, 1, index=True)
        _, col_dict_12, _ = self.complex.incidence_matrix(1, 2, index=True)
        
        # Use the column dictionary to get all 2-cells
        cells_2 = list(col_dict_12.keys())
        num_2_cells = len(cells_2)
        
        if num_2_cells == 0:
            print("No 2-cells found, creating dummy 2-cell features")
            return torch.zeros((num_samples, 1, self.feature_dim_2))
        
        print(f"Found {num_2_cells} 2-cells (PLC zones)")
        
        # Create tensor for 2-cell features
        x_2 = torch.zeros((num_samples, num_2_cells, self.feature_dim_2))
        
        # Get component to index mapping
        component_to_idx = {comp: i for i, comp in enumerate(self.columns)}
        
        # For each 2-cell, compute average features of its components
        for cell_idx, cell in enumerate(cells_2):
            try:
                # Extract 0-cells directly from the 2-cell frozenset
                node_set = set(cell)
                
                # Get indices of these 0-cells in the row dictionary
                node_indices = []
                for node in node_set:
                    node_frozenset = frozenset({node})
                    if node_frozenset in row_dict_01:
                        node_indices.append(row_dict_01[node_frozenset])
                
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

    def _compute_enhanced_x2(self, node_index_map):
        """
        Compute enhanced 2-cell features with 4D vectors:
        [sensor_mean, sensor_std, actuator_median, actuator_range]
        """
        print("Computing simplified 2-cell features (4D: mean/std for sensors, median/range for actuators)...")
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
            node_indices = [node_index_map.get(node) for node in node_set]
            node_indices = [idx for idx in node_indices if idx is not None]
            
            node_names = [self.columns[i] for i in node_indices]
            
            sensor_indices = [i for i, name in zip(node_indices, node_names) if not is_actuator("SWAT", name)]
            actuator_indices = [i for i, name in zip(node_indices, node_names) if is_actuator("SWAT", name)]
            cell_to_nodes[cell_idx] = (sensor_indices, actuator_indices)

        for sample_idx in range(num_samples):
            sample_x0 = self.x_0[sample_idx, :, 0]
            
            for cell_idx in range(num_2_cells):
                sensor_indices, actuator_indices = cell_to_nodes[cell_idx]
                
                simplified_feat = torch.zeros(self.feature_dim_2)
                
                # Sensor features: mean and std
                if sensor_indices:
                    sensor_features = sample_x0[sensor_indices]
                    simplified_feat[0] = torch.mean(sensor_features)
                    simplified_feat[1] = torch.std(sensor_features) if len(sensor_indices) > 1 else 0.0
                
                # Actuator features: median and range
                if actuator_indices:
                    actuator_features = sample_x0[actuator_indices]
                    simplified_feat[2] = torch.median(actuator_features)
                    simplified_feat[3] = torch.max(actuator_features) - torch.min(actuator_features)  # range
                
                x_2[sample_idx, cell_idx] = simplified_feat
        
        print(f"Computed simplified 2-cell features shape: {x_2.shape}")
        return x_2

    def create_adjacency_matrices(self):
        """Create adjacency matrices for the complex."""
        print("Creating adjacency matrices...")
        
        # Get topology matrices
        self.a0, self.a1, self.coa2, self.b1, self.b2 = self.get_neighborhood_matrix()
        
        print("Adjacency matrices created successfully")

    def get_neighborhood_matrix(self):
        """Get neighborhood matrices for the combinatorial complex."""
        # Get adjacency matrices
        a0 = self.complex.adjacency_matrix(rank=0, via_rank=1, index=False)
        a1 = self.complex.adjacency_matrix(rank=1, via_rank=2, index=False)
        coa2 = self.complex.coadjacency_matrix(rank=2, via_rank=0, index=False)
        
        # Get incidence matrices
        b1 = self.complex.incidence_matrix(rank=0, to_rank=1, index=False)
        b2 = self.complex.incidence_matrix(rank=1, to_rank=2, index=False)
        
        # Convert to sparse tensors (following pattern from other datasets)
        a0_tensor = torch.from_numpy(a0.toarray()).to_sparse()
        a1_tensor = torch.from_numpy(a1.toarray()).to_sparse()
        coa2_tensor = torch.from_numpy(coa2.toarray()).to_sparse()
        b1_tensor = torch.from_numpy(b1.toarray()).to_sparse()
        b2_tensor = torch.from_numpy(b2.toarray()).to_sparse()
        
        return a0_tensor, a1_tensor, coa2_tensor, b1_tensor, b2_tensor

    def create_temporal_samples(self):
        """Create temporal samples for temporal mode."""
        if not self.temporal_mode:
            return
            
        print(f"Creating temporal samples with sequence length {self.n_input}")
        
        # Apply temporal sampling if specified
        if self.temporal_sample_rate > 1:
            sampled_indices = list(range(0, len(self.data), self.temporal_sample_rate))
            self.data = self.data.iloc[sampled_indices].reset_index(drop=True)
            self.labels = self.labels[sampled_indices]
            print(f"Applied temporal sampling every {self.temporal_sample_rate} timesteps: {len(self.data)} samples remaining")
        
        # Calculate effective length for temporal mode
        self.effective_length = len(self.data) - self.n_input
        print(f"Effective samples in temporal mode: {self.effective_length}")

    def print_sample_features(self):
        """Print sample features for debugging."""
        print("\n=== Sample Features ===")
        print(f"Sample 0-cell features (first 3 components): {self.x_0[0, :3]}")
        print(f"Sample 1-cell features (first 3 edges): {self.x_1[0, :3]}")
        print(f"Sample 2-cell features (first 3 faces): {self.x_2[0, :3]}")
        print(f"Sample label: {self.labels[0]}")

    def __len__(self):
        return self.effective_length

    def __getitem__(self, idx):
        if self.temporal_mode:
            # Return temporal sequence
            seq_x0 = self.x_0[idx:idx + self.n_input]
            seq_x1 = self.x_1[idx:idx + self.n_input]
            seq_x2 = self.x_2[idx:idx + self.n_input]
            
            return {
                'x_0': seq_x0,
                'x_1': seq_x1,
                'x_2': seq_x2,
                'a0': self.a0,
                'a1': self.a1,
                'coa2': self.coa2,
                'b1': self.b1,
                'b2': self.b2,
                'label': self.labels[idx + self.n_input - 1]
            }
        else:
            # Return single timestep
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