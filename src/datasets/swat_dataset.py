import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


from src.utils.attack_utils import is_actuator


class SWaTDataset(Dataset):
    """Dataset class for SWAT anomaly detection."""
    def __init__(self, data, swat_complex, feature_dim_0=3, n_input=10, temporal_mode=False, temporal_sample_rate=1, use_geco_features=False):
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
        """
        self.data = data
        self.swat_complex = swat_complex  # Store the wrapper object
        self.complex = swat_complex.get_complex()  # Get the actual complex
        self.feature_dim_0 = feature_dim_0
        self.n_input = n_input
        self.temporal_mode = temporal_mode
        self.temporal_sample_rate = temporal_sample_rate
        self.use_geco_features = use_geco_features
        
        # Set feature dimensions based on GECO usage
        if use_geco_features:
            self.feature_dim_1 = feature_dim_0 * 2 + 2  # 6D concatenated + 2D GECO features = 8D
            print(f"Using GECO-enhanced edge features: {self.feature_dim_1}D")
        else:
            self.feature_dim_1 = feature_dim_0 * 2  # 6D concatenated features
            print(f"Using standard edge features: {self.feature_dim_1}D")
        
        self.feature_dim_2 = feature_dim_0  # 2-cells use same dimension as 0-cells
        
        # Process data and create features
        self.preprocess_features()
        self.x_0 = self.compute_initial_x0()
        self.x_1 = self.compute_initial_x1()
        self.x_2 = self.compute_initial_x2()
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

    def compute_initial_x0(self):
        """
        Preprocess raw features to create standardized 3D feature vectors.
        For sensors: [normalized_value, 0, 0]
        For actuators: one-hot encoded states, e.g., [0, 1, 0] for state 1,
                      with state 0 as [0, 0, 0] (neutral)
        """
        num_samples = len(self.data)
        num_components = len(self.columns)

        # Initialize tensor for 0-cell features
        features = torch.zeros((num_samples, num_components, self.feature_dim_0))

        print("Preprocessing features...")
        
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

    def compute_initial_x1(self):
        """
        Initialize 1-cell features based on connected 0-cells.
        Concatenate features and optionally add GECO features.

        Returns
        -------
        torch.Tensor
            Tensor of shape [num_timestamps, num_1_cells, feature_dim_1] 
            where feature_dim_1 = feature_dim_0 * 2 [+ 2 if using GECO features]
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
                        # Add simplified GECO features: [strength, equation_type]
                        equation_type = 1.0 if geco_features['geco_equation'] == 'Product' else 0.0
                        geco_feat = torch.tensor([
                            geco_features['geco_strength'],
                            equation_type
                        ], dtype=torch.float32)
                    else:
                        # Default GECO features for non-GECO edges
                        geco_feat = torch.tensor([0.0, 0.0], dtype=torch.float32)
                    
                    # Concatenate original features with GECO features
                    edge_feat = torch.cat([edge_feat, geco_feat], dim=0)  # 8D total feature
                
                x_1[sample_idx, new_cell_idx] = edge_feat

        print(f"Computed 1-cell features shape: {x_1.shape}")
        return x_1

    def compute_initial_x2(self):
        """
        Initialize 2-cell features based on average of connected 0-cells.
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