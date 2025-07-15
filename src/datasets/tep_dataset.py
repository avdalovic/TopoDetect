import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.attack_utils import is_actuator, TEP_COLUMN_NAMES

class TEPDataset(Dataset):
    """Dataset class for TEP (Tennessee Eastman Process) anomaly detection."""
    def __init__(self, data, tep_complex, feature_dim_0=3, n_input=10, temporal_mode=False, temporal_sample_rate=1):
        """
        Initialize the TEP dataset for anomaly detection.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing the TEP measurements and labels
        tep_complex : toponetx.CombinatorialComplex
            Combinatorial complex representing the TEP system topology
        feature_dim_0 : int, default=3
            Dimension of feature vectors for 0-cells (3 to accommodate one-hot encoding)
        n_input : int, default=10
            Number of input timesteps for temporal mode
        temporal_mode : bool, default=False
            Whether to use temporal sequences or single timesteps
        temporal_sample_rate : int, default=1
            Rate at which to subsample temporal data. 1 means no subsampling.
        """
        # Truncate data at 15000 points since attacks only go up to 14000
        if len(data) > 15000:
            print(f"TEP Dataset: Truncating data from {len(data)} to 15000 points (attacks end at 14000)")
            data = data.iloc[:15000].copy()
        
        self.data = data
        self.complex = tep_complex.get_complex()
        self.feature_dim_0 = feature_dim_0  # 0-cells: 3D
        self.feature_dim_1 = feature_dim_0 * 2  # 1-cells: 6D (concatenation of two 0-cells)
        self.feature_dim_2 = feature_dim_0 + 1  # 2-cells: 4D (avg of 0-cells + subprocess size)
        self.n_input = n_input
        self.temporal_mode = temporal_mode

        # Extract features and labels
        # TEP uses 'Atk' column for labels (0=normal, 1=attack)
        self.labels = data['Atk'].values.astype(int)

        # Get sensor and actuator columns (excluding attack label)
        self.columns = [col for col in data.columns if col != 'Atk']
        
        # Ensure we have the expected 53 components (52 sensors/actuators + 1 attack label)
        print(f"TEP Dataset: Found {len(self.columns)} components (expected 52)")
        print(f"Components: {self.columns[:10]}..." if len(self.columns) > 10 else f"Components: {self.columns}")

        # Preprocess features
        self.x_0 = self.preprocess_features()

        # Get topology matrices (only need to compute once since topology is fixed)
        self.a0, self.a1, self.coa2, self.b1, self.b2 = self.get_neighborhood_matrix()

        # Compute initial features for 1-cells and 2-cells
        self.x_1 = self.compute_initial_x1()
        self.x_2 = self.compute_initial_x2()

        # Calculate effective length based on temporal mode
        if self.temporal_mode and temporal_sample_rate > 1:
            # Subsample data to reduce training cost
            sampled_indices = list(range(0, len(self.data), temporal_sample_rate))
            self.data = self.data.iloc[sampled_indices].reset_index(drop=True)
            self.labels = self.labels[sampled_indices]
            print(f"Applied temporal sampling every {temporal_sample_rate} timesteps: {len(self.data)} samples remaining")

        if self.temporal_mode:
            self.effective_length = len(self.data) - self.n_input
            print(f"Using temporal mode with sequence length {self.n_input}, effective samples: {self.effective_length}")
        else:
            self.effective_length = len(self.data)

        print(f"Initialized TEPDataset with {self.effective_length} samples")
        print(f"Feature dimensions - 0-cells: {self.x_0.shape} (dim={self.feature_dim_0}), 1-cells: {self.x_1.shape} (dim={self.feature_dim_1}), 2-cells: {self.x_2.shape} (dim={self.feature_dim_2})")
        
        # Debug: Print sample features
        self.print_sample_features()

    def preprocess_features(self):
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

        print("Preprocessing TEP features...")
        
        # Track which indices are actuators for later
        actuator_indices = []
        
        for i, col in enumerate(self.columns):
            values = self.data[col].values

            if is_actuator("TEP", col):
                # For actuators: one-hot encode, but state 0 is neutral [0,0,0]
                actuator_indices.append(i)
                
                # TEP actuators are continuous values (0-100), so we need to discretize
                # or treat as continuous. For now, let's treat as continuous in first dimension
                for sample_idx in range(num_samples):
                    value = values[sample_idx]
                    # Normalize actuator values (typically 0-100 range)
                    normalized_value = value / 100.0
                    features[sample_idx, i, 0] = normalized_value
                    # Could add discrete encoding in dimensions 1,2 based on value ranges
                    if normalized_value > 0.66:
                        features[sample_idx, i, 2] = 1.0  # High
                    elif normalized_value > 0.33:
                        features[sample_idx, i, 1] = 1.0  # Medium
                    # Low remains [val, 0, 0]
            else:
                # For sensors: store value in first dimension
                features[:, i, 0] = torch.tensor(values, dtype=torch.float)

        # For sensors, standardize values (after storing them)
        sensor_indices = [i for i, col in enumerate(self.columns) if not is_actuator("TEP", col)]
        
        # Calculate mean and std for each sensor (across all time points)
        sensor_means = torch.zeros(len(sensor_indices))
        sensor_stds = torch.zeros(len(sensor_indices))
        
        for idx, i in enumerate(sensor_indices):
            values = features[:, i, 0]
            sensor_means[idx] = values.mean()
            sensor_stds[idx] = values.std() + 1e-6  # Add small epsilon to avoid division by zero
            
            # Standardize: (x - mean) / std
            features[:, i, 0] = (values - sensor_means[idx]) / sensor_stds[idx]
        
        print(f"Normalized {len(sensor_indices)} sensor features, mean: {sensor_means.mean():.4f}, std: {sensor_stds.mean():.4f}")
        print(f"Processed {len(actuator_indices)} actuator features with continuous + discrete encoding")
        
        print(f"Preprocessed features shape: {features.shape}")
        return features

    def get_neighborhood_matrix(self):
        """
        Compute the neighborhood matrices for the TEP complex.

        Returns
        -------
        tuple of torch.sparse.Tensor
            Adjacency, coadjacency and incidence matrices
        """
        print("Computing neighborhood matrices...")

        # Get adjacency matrices
        a0 = torch.from_numpy(self.complex.adjacency_matrix(0, 1).todense()).to_sparse()
        a1 = torch.from_numpy(self.complex.adjacency_matrix(1, 2).todense()).to_sparse()

        # Get incidence matrices
        b1 = torch.from_numpy(self.complex.incidence_matrix(0, 1).todense()).to_sparse()
        b2 = torch.from_numpy(self.complex.incidence_matrix(1, 2).todense()).to_sparse()

        # Compute coadjacency matrix for 2-cells
        B = self.complex.incidence_matrix(0, 2)
        A = B.T @ B
        A.setdiag(0)
        coa2 = torch.from_numpy(A.todense()).to_sparse()

        print(f"Matrix dimensions - a0: {a0.shape}, a1: {a1.shape}, coa2: {coa2.shape}, b1: {b1.shape}, b2: {b2.shape}")
        return a0, a1, coa2, b1, b2

    def compute_initial_x1(self):
        """
        Initialize 1-cell features based on connected 0-cells.
        Concatenate features instead of averaging to preserve information.

        Returns
        -------
        torch.Tensor
            Tensor of shape [num_timestamps, num_1_cells, feature_dim_1] where feature_dim_1 = feature_dim_0 * 2
        """
        print("Computing initial 1-cell features (concatenation method)...")
        num_samples = len(self.data)

        # Get row and column dictionaries for 0-1 incidence matrix
        row_dict, col_dict, _ = self.complex.incidence_matrix(0, 1, index=True)
        
        # Use the column dictionary to get all 1-cells
        cells_1 = list(col_dict.keys())
        num_1_cells = len(cells_1)
        
        print(f"Found {num_1_cells} 1-cells")

        # Create tensor for 1-cell features (now 6D due to concatenation)
        x_1 = torch.zeros((num_samples, num_1_cells, self.feature_dim_1))

        # For each 1-cell (edge), concatenate the features of its boundary 0-cells
        for cell_idx, cell in enumerate(cells_1):
            # Each 1-cell is a frozenset containing two 0-cells
            boundary_nodes = list(cell)
            if len(boundary_nodes) != 2:
                print(f"Warning: 1-cell {cell} has {len(boundary_nodes)} boundary nodes, expected 2")
                continue

            # Get indices of boundary nodes in the row dictionary
            try:
                node_indices = [row_dict[frozenset({node})] for node in boundary_nodes]

                # Concatenate the features of the boundary nodes
                for sample_idx in range(num_samples):
                    # Concatenate features of connected 0-cells
                    node1_feat = self.x_0[sample_idx, node_indices[0]]  # First 3 dimensions
                    node2_feat = self.x_0[sample_idx, node_indices[1]]  # Next 3 dimensions
                    edge_feat = torch.cat([node1_feat, node2_feat], dim=0)  # 6D concatenated feature
                    x_1[sample_idx, cell_idx] = edge_feat

            except KeyError as e:
                print(f"Warning: Couldn't find node in row_dict: {e}")

        print(f"Computed 1-cell features shape: {x_1.shape}")
        return x_1

    def compute_initial_x2(self):
        """
        Initialize 2-cell features based on grouped 0-cells.
        Add subprocess size information as 4th dimension.

        Returns
        -------
        torch.Tensor
            Tensor of shape [num_timestamps, num_2_cells, feature_dim_2] where feature_dim_2 = feature_dim_0 + 1
        """
        print("Computing initial 2-cell features (with subprocess size)...")
        num_samples = len(self.data)

        # Get row dictionaries for 0-1 and 1-2 incidence matrices
        row_dict_01, _, _ = self.complex.incidence_matrix(0, 1, index=True)
        _, col_dict_12, _ = self.complex.incidence_matrix(1, 2, index=True)
        
        # Use the column dictionary to get all 2-cells
        cells_2 = list(col_dict_12.keys())
        num_2_cells = len(cells_2)
        
        print(f"Found {num_2_cells} 2-cells")

        # Create tensor for 2-cell features (now 4D: 3D averaged + 1D subprocess size)
        x_2 = torch.zeros((num_samples, num_2_cells, self.feature_dim_2))

        # Total number of 0-cells for relative size calculation
        total_0_cells = len(self.columns)

        # For each 2-cell, collect and average the features of its 0-cells + add subprocess size
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
                
                # If we have valid nodes, average their features and add subprocess size
                if node_indices:
                    # Calculate relative subprocess size (number of 0-cells in this 2-cell / total 0-cells)
                    subprocess_size = len(node_indices) / total_0_cells
                    
                    for sample_idx in range(num_samples):
                        # Average features of all 0-cells in the group (first 3 dimensions)
                        face_feat = torch.zeros(self.feature_dim_0)
                        for node_idx in node_indices:
                            face_feat += self.x_0[sample_idx, node_idx]
                        face_feat /= len(node_indices)
                        
                        # Combine averaged features with subprocess size
                        full_feat = torch.cat([face_feat, torch.tensor([subprocess_size])], dim=0)
                        x_2[sample_idx, cell_idx] = full_feat
                else:
                    print(f"Warning: No valid 0-cells found for 2-cell {cell}")
                    # Set subprocess size to 0 if no valid nodes
                    x_2[:, cell_idx, -1] = 0.0

            except Exception as e:
                print(f"Warning: Error computing 2-cell features for cell {cell}: {e}")
                print(f"Cell contents: {cell}")

        print(f"Computed 2-cell features shape: {x_2.shape}")
        return x_2

    def print_sample_features(self):
        """Print sample features for debugging."""
        if len(self.data) == 0:
            return
            
        print("\n=== DEBUG: Sample TEP Features ===")
        
        # Print one 0-cell feature
        sample_0_cell = self.x_0[0, 0]  # First sample, first 0-cell
        print(f"Sample 0-cell feature (component '{self.columns[0]}'): {sample_0_cell.numpy()}")
        
        # Print one 1-cell feature
        if self.x_1.shape[1] > 0:
            sample_1_cell = self.x_1[0, 0]  # First sample, first 1-cell
            print(f"Sample 1-cell feature: {sample_1_cell.numpy()}")
        
        # Print one 2-cell feature
        if self.x_2.shape[1] > 0:
            sample_2_cell = self.x_2[0, 0]  # First sample, first 2-cell
            print(f"Sample 2-cell feature: {sample_2_cell.numpy()}")
            print(f"  - First 3 dims (averaged 0-cell features): {sample_2_cell[:3].numpy()}")
            print(f"  - 4th dim (subprocess size): {sample_2_cell[3].item():.4f}")
        
        print("=== END DEBUG ===\n")

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