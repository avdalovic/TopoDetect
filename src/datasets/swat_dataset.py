import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# The old import was: from utils.attack_utils import is_actuator
# The new import reflects the new 'src' structure
from src.utils.attack_utils import is_actuator


class SWaTDataset(Dataset):
    """Dataset class for SWAT anomaly detection."""
    def __init__(self, data, swat_complex, feature_dim=3, n_input=10, temporal_mode=False, temporal_sample_rate=1):
        """
        Initialize the SWAT dataset for anomaly detection.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing the SWAT measurements and labels
        swat_complex : toponetx.CombinatorialComplex
            Combinatorial complex representing the SWAT system topology
        feature_dim : int, default=3
            Dimension of feature vectors (3 to accommodate one-hot encoding)
        n_input : int, default=10
            Number of input timesteps for temporal mode
        temporal_mode : bool, default=False
            Whether to use temporal sequences or single timesteps
        temporal_sample_rate : int, default=1
            Rate at which to subsample temporal data. 1 means no subsampling.
        """
        self.data = data
        self.complex = swat_complex.get_complex()
        self.feature_dim = feature_dim
        self.n_input = n_input
        self.temporal_mode = temporal_mode

        # Extract features and labels
        self.labels = data['Normal/Attack'].map(lambda x: 0 if (x == 'False' or x == 0 or x == 0.0) else 1).values

        # Get sensor and actuator columns (excluding timestamp and label)
        self.columns = [col for col in data.columns if col not in ['Timestamp', 'Normal/Attack']]

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

        print(f"Initialized SWaTDataset with {self.effective_length} samples")
        print(f"Feature dimensions - 0-cells: {self.x_0.shape}, 1-cells: {self.x_1.shape}, 2-cells: {self.x_2.shape}")

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
        features = torch.zeros((num_samples, num_components, self.feature_dim))

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
        print(f"Actuator encoding: state 0=[0,0,0], state 1=[0,1,0], state 2=[0,0,1]")
        
        print(f"Preprocessed features shape: {features.shape}")
        return features

    def get_neighborhood_matrix(self):
        """
        Compute the neighborhood matrices for the SWAT complex.

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

        Returns
        -------
        torch.Tensor
            Tensor of shape [num_timestamps, num_1_cells, feature_dim]
        """
        print("Computing initial 1-cell features...")
        num_samples = len(self.data)

        # Get row and column dictionaries for 0-1 incidence matrix
        row_dict, col_dict, _ = self.complex.incidence_matrix(0, 1, index=True)
        
        # Use the column dictionary to get all 1-cells
        cells_1 = list(col_dict.keys())
        num_1_cells = len(cells_1)
        
        print(f"Found {num_1_cells} 1-cells")

        # Create tensor for 1-cell features
        x_1 = torch.zeros((num_samples, num_1_cells, self.feature_dim))

        # For each 1-cell (edge), average the features of its boundary 0-cells
        for cell_idx, cell in enumerate(cells_1):
            # Each 1-cell is a frozenset containing two 0-cells
            boundary_nodes = list(cell)
            if len(boundary_nodes) != 2:
                print(f"Warning: 1-cell {cell} has {len(boundary_nodes)} boundary nodes, expected 2")
                continue

            # Get indices of boundary nodes in the row dictionary
            try:
                node_indices = [row_dict[frozenset({node})] for node in boundary_nodes]

                # Average the features of the boundary nodes
                for sample_idx in range(num_samples):
                    # Average features of connected 0-cells
                    edge_feat = torch.zeros(self.feature_dim)
                    for node_idx in node_indices:
                        edge_feat += self.x_0[sample_idx, node_idx]
                    edge_feat /= len(node_indices)
                    x_1[sample_idx, cell_idx] = edge_feat

            except KeyError as e:
                print(f"Warning: Couldn't find node in row_dict: {e}")

        print(f"Computed 1-cell features shape: {x_1.shape}")
        return x_1

    def compute_initial_x2(self):
        """
        Initialize 2-cell features based on grouped 0-cells.

        Returns
        -------
        torch.Tensor
            Tensor of shape [num_timestamps, num_2_cells, feature_dim]
        """
        print("Computing initial 2-cell features...")
        num_samples = len(self.data)

        # Get row dictionaries for 0-1 and 1-2 incidence matrices
        row_dict_01, _, _ = self.complex.incidence_matrix(0, 1, index=True)
        _, col_dict_12, _ = self.complex.incidence_matrix(1, 2, index=True)
        
        # Use the column dictionary to get all 2-cells
        cells_2 = list(col_dict_12.keys())
        num_2_cells = len(cells_2)
        
        print(f"Found {num_2_cells} 2-cells")

        # Create tensor for 2-cell features
        x_2 = torch.zeros((num_samples, num_2_cells, self.feature_dim))

        # For each 2-cell, collect and average the features of its 0-cells
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
                        face_feat = torch.zeros(self.feature_dim)
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