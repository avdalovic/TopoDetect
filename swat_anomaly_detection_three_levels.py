"""
SWAT Anomaly Detection using Combinatorial Complex Attention Neural Network
This script implements an anomaly detection system for the SWAT dataset
using a topological neural network approach.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from topomodelx.nn.combinatorial.hmc import HMC
from utils.swat_topology import SWATComplex
from utils.attack_utils import get_attack_indices, is_actuator, get_attack_sds
import time
import pickle
import math
import random 

class SWaTDataset(Dataset):
    """Dataset class for SWAT anomaly detection."""
    def __init__(self, data, swat_complex, feature_dim=3, n_input=10, temporal_mode=False):
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
            
            # Label of the target timestep
            label = self.labels[idx+self.n_input]
            
            return {
                'seq_x0': seq_x0,
                'seq_x1': seq_x1,
                'seq_x2': seq_x2,
                'target_x0': target_x0,
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

class ResidualDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Main path
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # Residual projection (needed when input_dim != output_dim)
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    
    def forward(self, x):
        # Residual connection: output = F(x) + projection(x)
        return self.layers(x) + self.shortcut(x)

class AnomalyCCANN(nn.Module):
    """
    Anomaly detection model using Combinatorial Complex Attention Neural Network.
    Uses either an autoencoder approach to reconstruct features, or a temporal approach
    to predict the next timestep's features.
    """
    def __init__(self, channels_per_layer, original_feature_dim, temporal_mode=False, n_input=10):
        """
        Initialize the anomaly detection model.

        Parameters
        ----------
        channels_per_layer : list
            List of layer configurations for the HMC encoder
        original_feature_dim : int
            Original feature dimension for 0-cells
        temporal_mode : bool, default=False
            Whether to use temporal mode with LSTM encoding
        n_input : int, default=10
            Number of input timesteps in temporal mode
        """
        super().__init__()
        self.temporal_mode = temporal_mode
        self.n_input = n_input
        
        # Original HMC encoder (for message passing between cells)
        self.encoder = HMC(channels_per_layer)

        # Get the output dimension of the last layer for 0-cells
        final_dim = channels_per_layer[-1][2][0]
        self.final_dim = final_dim

        # LSTM for temporal encoding (only used in temporal mode)
        if temporal_mode:
            self.lstm = nn.LSTM(
                input_size=original_feature_dim,
                hidden_size=final_dim,
                num_layers=1,
                batch_first=True
            )
            
            # Variance projection for 2-cells
            self.variance_proj = nn.Linear(original_feature_dim, final_dim)
        
        # Decoder to reconstruct/predict 0-cell features
        self.decoder = ResidualDecoder(final_dim, 64, original_feature_dim)
        
        # Added for hierarchical localization: decoders for 1-cells and 2-cells
        self.decoder_1cell = ResidualDecoder(final_dim, 64, original_feature_dim)
        self.decoder_2cell = ResidualDecoder(final_dim, 64, original_feature_dim)
        
        # Optional decoder for 2-cell mean prediction
        if temporal_mode:
            self.decoder_2cell = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.ReLU(),
            nn.Linear(64, original_feature_dim)
        )

        print(f"Initialized AnomalyCCANN with encoder output dim {final_dim} and decoder output dim {original_feature_dim}")
        if temporal_mode:
            print(f"Using temporal mode with sequence length {n_input}")

    def forward(self, *args, **kwargs):
        """
        Forward pass through the model. Handles both temporal and non-temporal modes.
        """
        if self.temporal_mode:
            # If it's a dictionary input from the dataset's temporal mode
            if isinstance(args[0], dict):
                batch = args[0]
                return self.forward_temporal(
                    batch['seq_x0'], 
                    batch['seq_x1'], 
                    batch['seq_x2'], 
                    batch['a0'], 
                    batch['a1'], 
                    batch['coa2'], 
                    batch['b1'], 
                    batch['b2']
                )
            # If it's individual tensors as arguments
            else:
                return self.forward_temporal(*args)
        else:
            # Original non-temporal forward pass
            return self.forward_original(*args)

    def forward_original(self, x_0, x_1, x_2, a0, a1, coa2, b1, b2):
        """
        Original forward pass for reconstruction.
        
        Parameters
        ----------
        x_0 : torch.Tensor
            Features for 0-cells
        x_1 : torch.Tensor
            Features for 1-cells
        x_2 : torch.Tensor
            Features for 2-cells
        a0, a1, coa2, b1, b2 : torch.sparse.Tensor
            Topology matrices

        Returns
        -------
        torch.Tensor
            Reconstructed 0-cell features
        """
        # Debug print to see what we're getting
        # print(f"DEBUG: Input shapes - x_0: {x_0.shape}, x_1: {x_1.shape}, x_2: {x_2.shape}")
        
        # Remove batch dimension from feature tensors
        x_0_no_batch = x_0.squeeze(0)  # [num_0_cells, feature_dim]
        x_1_no_batch = x_1.squeeze(0)  # [num_1_cells, feature_dim]
        x_2_no_batch = x_2.squeeze(0)  # [num_2_cells, feature_dim]
        
        # For sparse tensors, handle batch dimension if present
        if a0.dim() > 2:
            # Move to CPU, convert to dense, squeeze, then back to sparse
            a0_no_batch = a0.to('cpu').to_dense().squeeze(0).to_sparse()
            a1_no_batch = a1.to('cpu').to_dense().squeeze(0).to_sparse()
            coa2_no_batch = coa2.to('cpu').to_dense().squeeze(0).to_sparse()
            b1_no_batch = b1.to('cpu').to_dense().squeeze(0).to_sparse()
            b2_no_batch = b2.to('cpu').to_dense().squeeze(0).to_sparse()
            
            # Move back to original device
            device = x_0.device
            a0_no_batch = a0_no_batch.to(device)
            a1_no_batch = a1_no_batch.to(device)
            coa2_no_batch = coa2_no_batch.to(device)
            b1_no_batch = b1_no_batch.to(device)
            b2_no_batch = b2_no_batch.to(device)
        else:
            # Already 2D, no need to squeeze
            a0_no_batch = a0
            a1_no_batch = a1
            coa2_no_batch = coa2
            b1_no_batch = b1
            b2_no_batch = b2
        
        # Encode features using HMC (without batch dimension)
        try:
            x_0_enc, x_1_enc, x_2_enc = self.encoder(x_0_no_batch, x_1_no_batch, x_2_no_batch, 
                                         a0_no_batch, a1_no_batch, coa2_no_batch, 
                                         b1_no_batch, b2_no_batch)
        except Exception as e:
            print(f"ERROR in encoder: {e}")
            raise
        
        # Decode to reconstruct 0-cell features
        x_0_recon_no_batch = self.decoder(x_0_enc)

        # hierarchical localization

        x_1_recon_no_batch = self.decoder_1cell(x_1_enc)
        x_2_recon_no_batch = self.decoder_2cell(x_2_enc)
        # Add back the batch dimension
        x_0_recon = x_0_recon_no_batch.unsqueeze(0)
        x_1_recon = x_1_recon_no_batch.unsqueeze(0)
        x_2_recon = x_2_recon_no_batch.unsqueeze(0)
        return x_0_recon, x_1_recon, x_2_recon
    
    def forward_temporal(self, seq_x0, seq_x1, seq_x2, a0, a1, coa2, b1, b2):
        """
        Temporal forward pass for next-step prediction.
        """
        # Get shapes and ensure batch size is 1
        device = seq_x0.device
        batch_size = 1
        n_input = seq_x0.shape[1]
        n_nodes = seq_x0.shape[2]
        feature_dim = seq_x0.shape[3]
        
        # Reshape for LSTM: [batch_size=1 * n_nodes, n_input, feature_dim]
        seq_x0_flat = seq_x0.reshape(-1, n_input, feature_dim)
        
        # Run LSTM to encode temporal patterns
        lstm_out, _ = self.lstm(seq_x0_flat)
        
        # Get last timestep's output: [batch_size=1 * n_nodes, final_dim]
        h_0 = lstm_out[:, -1]
        
        # Reshape back to [batch_size=1, n_nodes, final_dim]
        h_0 = h_0.reshape(batch_size, n_nodes, self.final_dim)
        
        # Calculate 2-cell variances over time
        # First, get the 0-cells in each 2-cell using the b1 and b2 matrices
        if b1.dim() > 2:
            # Convert to dense and remove batch dimension
            b1_dense = b1.to_dense().squeeze(0)
            b2_dense = b2.to_dense().squeeze(0)
        else:
            # Already 2D
            b1_dense = b1.to_dense()
            b2_dense = b2.to_dense()
        
        # Compute variance of 0-cells in each 2-cell
        n_2cells = 5  # Explicitly set to 5 for the 5 PLCs (2-cells)
        
        # Initialize 1-cell features
        h_1 = torch.zeros(batch_size, b1_dense.shape[1], self.final_dim, device=device)
        
        # Initialize 2-cell features 
        h_2 = torch.zeros(batch_size, n_2cells, self.final_dim, device=device)
        
        # For each 2-cell, compute mean and variance of its 0-cells
        for i in range(n_2cells):
            # Get 1-cells in this 2-cell
            cell1_indices = torch.where(b2_dense[:, i] > 0)[0]
            
            # Get 0-cells connected to these 1-cells
            cell0_indices = set()
            for idx in cell1_indices:
                nodes = torch.where(b1_dense[:, idx] > 0)[0]
                for node in nodes:
                    cell0_indices.add(node.item())
            
            if cell0_indices:
                cell0_indices = list(cell0_indices)
                
                # Calculate variance across time for each 0-cell
                cell_vars = torch.var(seq_x0[:, :, cell0_indices, :], dim=1)  # [batch_size, n_nodes_in_2cell, feat_dim]
                mean_var = torch.mean(cell_vars, dim=1)  # [batch_size, feat_dim]
                
                # Project variance to embedding dimension
                var_proj = self.variance_proj(mean_var)  # [batch_size, final_dim]
                
                # For 2-cell features, use the mean of the LSTM outputs plus variance
                h_0_mean = torch.mean(h_0[:, cell0_indices, :], dim=1)  # [batch_size, final_dim]
                h_2[:, i, :] = h_0_mean + var_proj
        
        # For 1-cell features, use the product of connected 0-cell LSTM outputs
        for i in range(b1_dense.shape[1]):
            cell0_indices = torch.where(b1_dense[:, i] > 0)[0]
            if len(cell0_indices) == 2:
                h_1[:, i, :] = h_0[:, cell0_indices[0], :] * h_0[:, cell0_indices[1], :]
        
        # Process through HMC for message passing between different ranks
        # ENSURE WE HAVE PROPER 2D SPARSE TENSORS - THIS IS THE KEY FIX
        b1_no_batch = b1_dense.to_sparse()  # Now it's guaranteed to be 2D
        b2_no_batch = b2_dense.to_sparse()  # Now it's guaranteed to be 2D
        
        # Handle other sparse matrices
        if a0.dim() > 2:
            # Ensure a0, a1, coa2 are also proper 2D sparse tensors
            a0_no_batch = a0.to_dense().squeeze(0).to_sparse()
            a1_no_batch = a1.to_dense().squeeze(0).to_sparse()
            coa2_no_batch = coa2.to_dense().squeeze(0).to_sparse()
        else:
            # Already 2D
            a0_no_batch = a0
            a1_no_batch = a1
            coa2_no_batch = coa2
        
        # Process each item in the batch (just one item since batch_size=1)
        try:
            # Remove batch dimension for HMC
            x_0_enc, _, x_2_enc = self.encoder(
                h_0[0], h_1[0], h_2[0],
                a0_no_batch, a1_no_batch, coa2_no_batch, 
                b1_no_batch, b2_no_batch
            )
            
            # Predict next timestep features for 0-cells
            x0_pred = self.decoder(x_0_enc)
            
            # Predict mean for 2-cells (optional)
            x2_mean_pred = self.decoder_2cell(x_2_enc)
            
            # Add batch dimension back
            x0_preds = x0_pred.unsqueeze(0)
            x2_means_preds = x2_mean_pred.unsqueeze(0)
            
        except Exception as e:
            print(f"ERROR in encoder: {e}")
            print(f"DEBUG - h_0 shape: {h_0.shape}, h_1 shape: {h_1.shape}, h_2 shape: {h_2.shape}")
            print(f"DEBUG - h_0[0] shape: {h_0[0].shape}, h_1[0] shape: {h_1[0].shape}, h_2[0] shape: {h_2[0].shape}")
            print(f"DEBUG - a0 dims: {a0_no_batch.dim()}, b1 dims: {b1_no_batch.dim()}")
            raise
        
        return x0_preds, x2_means_preds

class AnomalyTrainer:
    """
    Trainer for the anomaly detection model.
    Handles both reconstruction-based and prediction-based approaches.
    """
    def __init__(self, model, train_dataloader, validation_dataloader, test_dataloader,
                 learning_rate, device, 
                 threshold_percentile=99.95,
                 use_geometric_mean=False, epsilon=1e-6,
                 threshold_method="percentile",
                 sd_multiplier=2.5, 
                 use_component_thresholds=True,
                 temporal_consistency=1,
                 weight_decay=1e-5,  # Add weight_decay parameter
                 grad_clip_value=1.0):  # Add grad_clip_value parameter
        """
        Initialize the trainer with additional training options.
        
        Additional Parameters
        --------------------
        weight_decay : float, default=1e-5
            L2 regularization factor for optimizer
        grad_clip_value : float, default=1.0
            Maximum norm for gradient clipping
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader # Store validation dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.threshold_percentile = threshold_percentile
        self.crit = nn.MSELoss(reduction='none')
        self.opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.anomaly_threshold = None
        self.cell2_threshold = None
        # New: Component-specific thresholds
        self.component_thresholds = None
        self.cell2_component_thresholds = None
        
        self.last_loss = None
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='min', factor=0.5, patience=3, verbose=True
        )
        self.use_geometric_mean = use_geometric_mean
        self.epsilon = epsilon
        self.threshold_method = threshold_method
        self.sd_multiplier = sd_multiplier 
        self.use_component_thresholds = use_component_thresholds
        self.temporal_consistency = temporal_consistency
        
        # Store new parameters
        self.grad_clip_value = grad_clip_value
        
        # Input validation for thresholding
        if self.validation_dataloader is None or len(self.validation_dataloader.dataset) == 0:
             raise ValueError("Validation dataloader is required and cannot be empty for threshold calibration.")
        if self.threshold_method == "percentile":
             print(f"Threshold method: percentile ({self.threshold_percentile}%)")
        elif self.threshold_method == "mean_sd":
             print(f"Threshold method: mean + {self.sd_multiplier} * SD")
        else:
             raise ValueError(f"Unknown threshold_method: {self.threshold_method}")


        print(f"Initialized AnomalyTrainer with device {device}")
        print(f"Using geometric mean for anomaly detection: {use_geometric_mean}")
        print(f"Using component-specific thresholds: {use_component_thresholds}")
        print(f"Temporal consistency requirement: {temporal_consistency} consecutive anomalies")
        print(f"Temporal mode: {model.temporal_mode}")
        print(f"Thresholds will be calibrated on validation data ({len(self.validation_dataloader.dataset)} samples).")

    def to_device(self, x):
        """
        Move data to device, handling both dictionary and list inputs.
        """
        if isinstance(x, dict):
            return {k: v.float().to(self.device) if isinstance(v, torch.Tensor) else v 
                   for k, v in x.items()}
        else:
            return [el.float().to(self.device) for el in x]

    def train_epoch(self):
        """
        Train the model for one epoch, supporting both temporal and non-temporal modes.
        With batch size fixed to 1 for HMC compatibility.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Get total number of samples to track progress
        total_samples = len(self.train_dataloader)
        print(f"Training on {total_samples} samples with batch size 1...")
        
        # Progress tracking
        log_interval = max(1, total_samples // 10)  # Show progress ~10 times per epoch
        start_time = time.time()

        for i, sample in enumerate(self.train_dataloader):
            # Zero gradients
            self.opt.zero_grad()

            # Handle different input formats based on temporal mode
            if self.model.temporal_mode:
                # Move sample dictionary to device
                sample = self.to_device(sample)
                
                # Forward pass - returns tuple of (x0_pred, x2_mean_pred)
                x0_pred, x2_mean_pred = self.model(sample)
                
                # Target for 0-cells (batch size is guaranteed to be 1)
                target_x0 = sample['target_x0']
                
                # Compute loss for 0-cell predictions
                loss_0 = self.crit(x0_pred, target_x0).mean()
                
                # Compute 2-cell mean targets and loss
                # Extract 0-cells in each 2-cell using b1 and b2 matrices
                if sample['b1'].dim() > 2:
                    b1 = sample['b1'].to_dense().squeeze(0)
                    b2 = sample['b2'].to_dense().squeeze(0)
                else:
                    b1 = sample['b1'].to_dense()
                    b2 = sample['b2'].to_dense()
                
                # Find all 0-cells for each 2-cell
                cell2_to_cell0 = {}
                
                # Use the correct dimension for n_2cells: 5 PLCs (2-cells)
                n_2cells = 5  # Explicitly set to 5 for the 5 PLCs (2-cells)
                
                for face_idx in range(n_2cells):
                    # Get 1-cells in this 2-cell
                    cell1_indices = torch.where(b2[:, face_idx] > 0)[0]
                    
                    # Get 0-cells connected to these 1-cells
                    cell0_indices = set()
                    for idx in cell1_indices:
                        nodes = torch.where(b1[:, idx] > 0)[0]
                        for node in nodes:
                            cell0_indices.add(node.item())
                    
                    cell2_to_cell0[face_idx] = list(cell0_indices)
                
                # Compute mean target for each 2-cell
                batch_size = target_x0.shape[0]
                feature_dim = target_x0.shape[2]
                
                # Initialize tensor for 2-cell mean targets with correct shape [1, 5, 3]
                x2_mean_target = torch.zeros((batch_size, n_2cells, feature_dim), device=self.device)
                
                for face_idx, cell0_indices in cell2_to_cell0.items():
                    if cell0_indices:
                        # Mean of all 0-cells in this 2-cell
                        x2_mean_target[:, face_idx] = torch.mean(target_x0[:, cell0_indices], dim=1)
                
                # Compute error for 2-cell mean predictions
                loss_2 = self.crit(x2_mean_pred, x2_mean_target).mean()
                
                # Combined loss (with weight for 2-cell loss)
                loss = loss_0 + 0.5 * loss_2
                
                # For logging
                total_loss += loss_0.item()  # Track only 0-cell loss for consistency
            else:
                # Original non-temporal processing
                x_0, x_1, x_2, a0, a1, coa2, b1, b2, _ = self.to_device(sample)
                
                # Forward pass (now returns 3 reconstructions)
                x_0_recon, x_1_recon, x_2_recon = self.model(x_0, x_1, x_2, a0, a1, coa2, b1, b2)
                
                # Compute loss for each level (0-cells, 1-cells, 2-cells)
                loss_0 = self.crit(x_0_recon, x_0).mean()
                loss_1 = self.crit(x_1_recon, x_1).mean()
                loss_2 = self.crit(x_2_recon, x_2).mean()
                
                # Combined loss with equal weights
                loss = 1/51 * loss_0 + 1/74 * loss_1 + 1/5 * loss_2
                
                # For logging
                total_loss += loss_0.item() # Keep tracking only 0-cell loss for consistency

            # Backward pass and optimize
            loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_value)  # Clip gradients to max norm self.grad_clip_value
            self.opt.step()
            
            num_batches += 1
            
            # Show progress at regular intervals
            if (i + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                samples_per_sec = (i + 1) / elapsed
                remaining = (total_samples - i - 1) / samples_per_sec if samples_per_sec > 0 else 0
                current_loss = total_loss / num_batches
                print(f"Progress: [{i+1}/{total_samples}] ({(i+1)/total_samples*100:.1f}%) | "
                      f"Loss: {current_loss:.6f} | "
                      f"Samples/sec: {samples_per_sec:.1f} | "
                      f"Time elapsed: {elapsed:.0f}s | "
                      f"Time remaining: {remaining:.0f}s")

            # Debug print to monitor gradient norms (optional)
            if i % log_interval == 0:
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"Gradient norm: {total_norm:.4f}")

        self.last_loss = total_loss / num_batches
        return self.last_loss

    def calculate_geometric_mean_error(self, errors):
        """
        Calculate geometric mean of errors.
        
        Parameters
        ----------
        errors : torch.Tensor
            Tensor of feature-wise errors
            
        Returns
        -------
        torch.Tensor
            Geometric mean of errors
        """
        # Add small epsilon to avoid zero errors and taking log of zero
        errors = errors + self.epsilon
        
        # Calculate geometric mean: exp(mean(log(errors)))
        log_errors = torch.log(errors)
        mean_log_errors = torch.mean(log_errors, dim=-1)
        geo_mean_errors = torch.exp(mean_log_errors)
        
        return geo_mean_errors

    def calibrate_threshold(self):
        """
        Calibrate anomaly detection thresholds based on the VALIDATION data.
        Supports component-specific thresholds.
        """
        print(f"\n--- Calibrating Thresholds on Validation Data ({len(self.validation_dataloader.dataset)} samples) ---")
        if self.threshold_method == "percentile":
            print(f"Using {self.threshold_percentile}th percentile...")
        else:
            print(f"Using mean+{self.sd_multiplier}×SD method...")
            
        error_type = "geometric mean" if self.use_geometric_mean else "mean"
        print(f"Using {error_type} of feature-wise errors")
        
        self.model.eval()
        all_0cell_errors = []
        all_1cell_errors = []  # Added for hierarchical localization
        all_2cell_errors = []  # Added for hierarchical localization
        
        # For component-specific thresholds
        component_errors_list = []  # Always collect for potential use
        group_errors_list = []      # Always collect for potential use

        # Add debug prints to see if validation data is loading
        print(f"Validation dataloader has {len(self.validation_dataloader)} batches")
        print(f"First validation sample shape: {next(iter(self.validation_dataloader))[0].shape}")
        
        # For debugging purposes, make error collection more robust with try/except
        with torch.no_grad():
            for sample_idx, sample in enumerate(self.validation_dataloader):
                try:
                    if self.model.temporal_mode:
                        # Move sample dictionary to device
                        sample = self.to_device(sample)
                        
                        # Forward pass - returns tuple of (x0_pred, x2_mean_pred)
                        x0_pred, x2_mean_pred = self.model(sample)
                        
                        # Target for 0-cells
                        target_x0 = sample['target_x0']
                        
                        # Compute error for 0-cell predictions
                        error_0 = self.crit(x0_pred, target_x0)  # [batch_size, num_components, feature_dim]
                        
                        # --- Store errors for threshold calculation ---
                        if self.use_geometric_mean:
                            # Calculate geometric mean across features
                            component_errors = self.calculate_geometric_mean_error(error_0)  # [batch_size, num_components]
                            component_errors_list.append(component_errors.cpu().numpy())
                            # For global threshold, average across components
                            sample_errors = torch.mean(component_errors, dim=1)  # [batch_size]
                            all_0cell_errors.append(sample_errors.cpu().numpy())
                        else:
                            # Original approach: mean across features
                            component_errors = error_0.mean(dim=2)  # [batch_size, num_components]
                            component_errors_list.append(component_errors.cpu().numpy())
                            # For global threshold, use max (or mean, depending on original intent)
                            sample_errors = component_errors.max(dim=1)[0] # Using max as before - change to mean if needed
                            all_0cell_errors.append(sample_errors.cpu().numpy())
                        # --- End Store errors ---

                        # --- Process 2-cell errors ---
                        if x2_mean_pred is not None: # Only if model outputs 2-cell predictions
                            b1 = sample['b1'].to_dense()
                            b2 = sample['b2'].to_dense()
                            
                            cell2_to_cell0 = {}
                            n_2cells = 5 # Explicitly set
                            
                            for face_idx in range(n_2cells):
                                cell1_indices = torch.where(b2[:, face_idx] > 0)[0]
                                cell0_indices = set()
                                for idx in cell1_indices:
                                    nodes = torch.where(b1[:, idx] > 0)[0]
                                    for node in nodes:
                                        cell0_indices.add(node.item())
                                cell2_to_cell0[face_idx] = list(cell0_indices)
                            
                            batch_size = target_x0.shape[0]
                            feature_dim = target_x0.shape[2]
                            x2_mean_target = torch.zeros((batch_size, n_2cells, feature_dim), device=self.device)
                            
                            for face_idx, cell0_indices in cell2_to_cell0.items():
                                if cell0_indices:
                                    x2_mean_target[:, face_idx] = torch.mean(target_x0[:, cell0_indices], dim=1)
                            
                            error_2 = self.crit(x2_mean_pred, x2_mean_target)
                            
                            if self.use_geometric_mean:
                                group_errors = self.calculate_geometric_mean_error(error_2)
                            else:
                                group_errors = error_2.mean(dim=2)
                            
                            group_errors_list.append(group_errors.cpu().numpy())
                            all_2cell_errors.append(torch.mean(group_errors, dim=1).cpu().numpy())
                         # --- End Process 2-cell errors ---

                        # Process 1-cell errors (new)
                        if self.use_geometric_mean:
                            edge_errors = self.calculate_geometric_mean_error(error_1)
                            edge_sample_errors = torch.mean(edge_errors, dim=1)
                            all_1cell_errors.append(edge_sample_errors.cpu().numpy())
                        else:
                            edge_errors = error_1.mean(dim=2)
                            edge_sample_errors = edge_errors.max(dim=1)[0]
                            all_1cell_errors.append(edge_sample_errors.cpu().numpy())
                    
                    else:
                        # Original non-temporal processing
                        x_0, x_1, x_2, a0, a1, coa2, b1, b2, _ = self.to_device(sample) # Only need x_0 for reconstruction target
                        
                        # Pass features and matrices - now returns a tuple of (x0_recon, x1_recon, x2_recon)
                        x0_recon_tuple = self.model(*self.to_device(sample[:-1]))
                        
                        # Unpack the returned values
                        x_0_recon, x_1_recon, x_2_recon = x0_recon_tuple
                        
                        # Compute errors for all three levels
                        error_0 = self.crit(x_0_recon, x_0)  # 0-cell errors
                        error_1 = self.crit(x_1_recon, x_1)  # 1-cell errors 
                        error_2 = self.crit(x_2_recon, x_2)  # 2-cell errors
                        
                        # ADD THIS CODE HERE to process and store errors
                        # Process 0-cell errors
                        if self.use_geometric_mean:
                            component_errors = self.calculate_geometric_mean_error(error_0)
                            component_errors_list.append(component_errors.cpu().numpy())
                            sample_errors = torch.mean(component_errors, dim=1)
                            all_0cell_errors.append(sample_errors.cpu().numpy())
                        else:
                            component_errors = error_0.mean(dim=2)
                            component_errors_list.append(component_errors.cpu().numpy())
                            sample_errors = component_errors.max(dim=1)[0]
                            all_0cell_errors.append(sample_errors.cpu().numpy())
                        
                        # Process 1-cell errors
                        if self.use_geometric_mean:
                            edge_errors = self.calculate_geometric_mean_error(error_1)
                            edge_sample_errors = torch.mean(edge_errors, dim=1)
                        else:
                            edge_errors = error_1.mean(dim=2)
                            edge_sample_errors = edge_errors.max(dim=1)[0]
                        all_1cell_errors.append(edge_sample_errors.cpu().numpy())
                        
                        # Process 2-cell errors
                        if self.use_geometric_mean:
                            cell2_errors = self.calculate_geometric_mean_error(error_2)
                            cell2_sample_errors = torch.mean(cell2_errors, dim=1)
                        else:
                            cell2_errors = error_2.mean(dim=2)
                            cell2_sample_errors = cell2_errors.max(dim=1)[0]
                        all_2cell_errors.append(cell2_sample_errors.cpu().numpy())
                
                
                    
                except Exception as e:
                    print(f"Error processing validation sample {sample_idx}: {e}")
                    print(f"Sample types: {[type(x) for x in sample]}")
                    # Continue to next sample instead of breaking out of the loop
                    continue
        
        # --- Calculate thresholds based on collected VALIDATION errors ---
        if not all_0cell_errors:
             print("Warning: No errors collected during threshold calibration. Cannot set thresholds.")
             return

        # Concatenate all errors for global threshold
        all_0cell_errors = np.concatenate(all_0cell_errors, axis=0)
        all_1cell_errors = np.concatenate(all_1cell_errors, axis=0)
        all_2cell_errors = np.concatenate(all_2cell_errors, axis=0)
        
        # Calculate global threshold based on selected method
        if self.threshold_method == "percentile":
            self.anomaly_threshold = np.percentile(all_0cell_errors, self.threshold_percentile)
            print(f"Calibrated global 0-cell anomaly threshold ({self.threshold_percentile}th percentile): {self.anomaly_threshold:.6f}")
            
            # Set 1-cell threshold
            self.cell1_threshold = np.percentile(all_1cell_errors, self.threshold_percentile)
            print(f"Calibrated global 1-cell anomaly threshold ({self.threshold_percentile}th percentile): {self.cell1_threshold:.6f}")
            
            # Set 2-cell threshold
            self.cell2_threshold = np.percentile(all_2cell_errors, self.threshold_percentile)
            print(f"Calibrated global 2-cell anomaly threshold ({self.threshold_percentile}th percentile): {self.cell2_threshold:.6f}")
        else:  # mean_sd method
            error_mean = np.mean(all_0cell_errors)
            error_std = np.std(all_0cell_errors)
            self.anomaly_threshold = error_mean + self.sd_multiplier * error_std
            print(f"Calibrated global 0-cell anomaly threshold (mean+{self.sd_multiplier}×SD): {self.anomaly_threshold:.6f}")
            print(f"Error distribution - Mean: {error_mean:.6f}, SD: {error_std:.6f}")
        
        # Calculate component-specific thresholds if enabled
        if self.use_component_thresholds and component_errors_list:
            component_errors_array = np.concatenate(component_errors_list, axis=0)
            # Compute per-component thresholds
            self.component_thresholds = []
            
            n_components = component_errors_array.shape[1]
            print(f"Calculating component-specific thresholds for {n_components} components...")
            
            for i in range(n_components):
                component_error = component_errors_array[:, i]
                if self.threshold_method == "percentile":
                    threshold = np.percentile(component_error, self.threshold_percentile)
                else:  # mean_sd method
                    mean = np.mean(component_error)
                    std = np.std(component_error)
                    threshold = mean + self.sd_multiplier * std
                self.component_thresholds.append(threshold)
            
            print(f"Component thresholds - min: {min(self.component_thresholds):.6f}, "
                  f"max: {max(self.component_thresholds):.6f}, "
                  f"mean: {np.mean(self.component_thresholds):.6f}")
        
        # Calculate 2-cell thresholds if applicable
        if self.model.temporal_mode and all_2cell_errors:
            all_2cell_errors = np.concatenate(all_2cell_errors, axis=0)
            
            # Global 2-cell threshold
            if self.threshold_method == "percentile":
                self.cell2_threshold = np.percentile(all_2cell_errors, self.threshold_percentile)
                print(f"Calibrated global 2-cell anomaly threshold ({self.threshold_percentile}th percentile): {self.cell2_threshold:.6f}")
            else:  # mean_sd method
                error2_mean = np.mean(all_2cell_errors)
                error2_std = np.std(all_2cell_errors)
                self.cell2_threshold = error2_mean + self.sd_multiplier * error2_std
                print(f"Calibrated global 2-cell anomaly threshold (mean+{self.sd_multiplier}×SD): {self.cell2_threshold:.6f}")
                print(f"2-cell error distribution - Mean: {error2_mean:.6f}, SD: {error2_std:.6f}")
            
            # Component-specific 2-cell thresholds if enabled
            if self.use_component_thresholds and group_errors_list:
                group_errors_array = np.concatenate(group_errors_list, axis=0)
                self.cell2_component_thresholds = []
                
                n_groups = group_errors_array.shape[1]
                print(f"Calculating component-specific 2-cell thresholds for {n_groups} groups...")
                
                for i in range(n_groups):
                    group_error = group_errors_array[:, i]
                    if self.threshold_method == "percentile":
                        threshold = np.percentile(group_error, self.threshold_percentile)
                    else:  # mean_sd method
                        mean = np.mean(group_error)
                        std = np.std(group_error)
                        threshold = mean + self.sd_multiplier * std
                    self.cell2_component_thresholds.append(threshold)
                
                print(f"2-cell thresholds - min: {min(self.cell2_component_thresholds):.6f}, "
                      f"max: {max(self.cell2_component_thresholds):.6f}, "
                      f"mean: {np.mean(self.cell2_component_thresholds):.6f}")
        print("--- Threshold Calibration Complete ---")

    def evaluate(self, dataloader=None, eval_mode="Test", component_names=None):
        """
        Evaluate the model on a given dataloader (defaults to test data).
        Uses component-specific thresholds and temporal consistency check if enabled.

        Parameters:
        ----------
        dataloader : DataLoader, optional
            The dataloader to evaluate on. If None, uses self.test_dataloader.
        eval_mode : str, default="Test"
            Label for evaluation output (e.g., "Validation", "Test")
        """
        if dataloader is None:
            dataloader = self.test_dataloader
            
        if self.anomaly_threshold is None:
             print(f"Warning: Threshold not calibrated before {eval_mode} evaluation. Calibrating now...")
             self.calibrate_threshold()
             if self.anomaly_threshold is None: # Check again after trying calibration
                 print(f"Error: Threshold calibration failed. Cannot proceed with {eval_mode} evaluation.")
                 return {'precision': 0, 'recall': 0, 'f1': 0, 'component_errors': [], 'predictions': [], 'labels': [], 'group_errors': None}


        print(f"\n--- Evaluating Model on {eval_mode} Data ({len(dataloader.dataset)} samples) ---")
        error_type = "geometric mean" if self.use_geometric_mean else "mean"
        print(f"Using {error_type} of feature-wise errors")
        if self.use_component_thresholds:
            print(f"Using component-specific thresholds")
            if not self.component_thresholds:
                 print("Warning: Component thresholds enabled but not calculated.")
        if self.temporal_consistency > 1:
            print(f"Requiring {self.temporal_consistency} consecutive anomalies for detection")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        component_errors = []  # 0-cell errors
        edge_errors = []       # Added for 1-cell errors
        cell2_errors = []      # Added for 2-cell errors
        anomaly_counts = {}    # Added missing initialization for temporal consistency
        
        preds_0cell = []
        preds_1cell = []
        preds_2cell = []
        preds_hierarchical = []
        anomalous_samples = {}  # Store anomalous cell details for localization
        
        with torch.no_grad():
             # *** CRITICAL CHANGE: Use the passed dataloader ***
            for sample_idx, sample in enumerate(dataloader): 
                if self.model.temporal_mode:
                    sample = self.to_device(sample)
                    x0_pred, x2_mean_pred = self.model(sample)
                    target_x0 = sample['target_x0']
                    label = sample['label']
                    error_0_raw = self.crit(x0_pred, target_x0)
                else:
                    # Non-temporal evaluation
                    sample_device = self.to_device(sample)
                    x_0 = sample_device[0]
                    x_1 = sample_device[1]  # Added to get 1-cells
                    x_2 = sample_device[2]  # Added to get 2-cells
                    label = sample_device[-1].item()
                    
                    # Pass features and matrices, exclude label
                    x0_pred, x1_pred, x2_pred = self.model(*sample_device[:-1])
                    
                    # Compute errors for all levels
                    error_0_raw = self.crit(x0_pred, x_0)
                    error_1_raw = self.crit(x1_pred, x_1)  # Added for 1-cells
                    error_2_raw = self.crit(x2_pred, x_2)  # Added for 2-cells
                    
                    # Process 0-cell errors with original logic
                    if self.use_geometric_mean:
                        error_0 = self.calculate_geometric_mean_error(error_0_raw)
                    else:
                        error_0 = error_0_raw.mean(dim=2)
                    
                    # Process 1-cell errors (similar pattern)
                    if self.use_geometric_mean:
                        error_1 = self.calculate_geometric_mean_error(error_1_raw)
                    else:
                        error_1 = error_1_raw.mean(dim=2)
                    
                    # Process 2-cell errors (similar pattern)
                    if self.use_geometric_mean:
                        error_2 = self.calculate_geometric_mean_error(error_2_raw)
                    else:
                        error_2 = error_2_raw.mean(dim=2)
                    
                    # Apply anomaly detection logic for each cell type and hierarchical approach
                    # Store cell-specific anomalies for localization
                    anomalous_cells = {
                        'cell0': [],
                        'cell1': [],
                        'cell2': []
                    }
                    
                    # 0-cell based prediction (existing logic)
                    is_anomalous_0cell = False
                    if self.use_component_thresholds and self.component_thresholds:
                        # Check component-specific thresholds
                        component_flags = error_0[0] > torch.tensor(self.component_thresholds, device=self.device)
                        is_anomalous_0cell = torch.any(component_flags).item()
                        
                        # Track which 0-cells are anomalous for localization
                        if is_anomalous_0cell:
                            for j in range(len(component_flags)):
                                if component_flags[j].item():
                                    anomalous_cells['cell0'].append({
                                        'id': j,  # Will be mapped to component name later
                                        'error': float(error_0[0, j]),
                                        'threshold': float(self.component_thresholds[j])
                                    })
                    else:
                        # Global threshold for 0-cells
                        for j in range(error_0.shape[1]):
                            cell_error = error_0[0, j]
                            if cell_error > self.anomaly_threshold:
                                is_anomalous_0cell = True
                                anomalous_cells['cell0'].append({
                                    'id': j,
                                    'error': float(cell_error),
                                    'threshold': float(self.anomaly_threshold)
                                })
                    
                    # 1-cell based prediction
                    is_anomalous_1cell = False
                    for j in range(error_1.shape[1]):
                        cell_error = error_1[0, j]
                        if cell_error > self.cell1_threshold:
                            is_anomalous_1cell = True
                            anomalous_cells['cell1'].append({
                                'id': j,
                                'error': float(cell_error),
                                'threshold': float(self.cell1_threshold)
                            })
                    
                    # 2-cell based prediction
                    is_anomalous_2cell = False
                    for j in range(error_2.shape[1]):
                        cell_error = error_2[0, j]
                        if cell_error > self.cell2_threshold:
                            is_anomalous_2cell = True
                            anomalous_cells['cell2'].append({
                                'id': j,
                                'error': float(cell_error),
                                'threshold': float(self.cell2_threshold)
                            })
                    
                    # Generate predictions for each approach
                    pred_0cell = 1 if is_anomalous_0cell else 0
                    pred_1cell = 1 if is_anomalous_1cell else 0
                    pred_2cell = 1 if is_anomalous_2cell else 0
                    
                    # Hierarchical prediction (2-cell > 1-cell > 0-cell)
                    if is_anomalous_2cell:
                        pred_hierarchical = 1
                    elif is_anomalous_1cell:
                        pred_hierarchical = 1
                    elif is_anomalous_0cell:
                        pred_hierarchical = 1
                    else:
                        pred_hierarchical = 0
                    
                    # Apply temporal consistency if enabled (using hierarchical prediction)
                    final_prediction = 0
                    if self.temporal_consistency > 1:
                        if pred_hierarchical == 1:
                            anomaly_counts[sample_idx] = anomaly_counts.get(sample_idx-1, 0) + 1
                        else:
                            anomaly_counts[sample_idx] = 0
                        if anomaly_counts[sample_idx] >= self.temporal_consistency:
                            final_prediction = 1
                    else:
                        final_prediction = pred_hierarchical
                    
                    # Store all predictions and errors
                    all_preds.append(final_prediction)
                    all_labels.append(label)
                    component_errors.append(error_0.cpu().numpy())
                    edge_errors.append(error_1.cpu().numpy())
                    cell2_errors.append(error_2.cpu().numpy())
                    
                    # Store per-approach predictions for separate metrics
                    preds_0cell.append(pred_0cell)
                    preds_1cell.append(pred_1cell)
                    preds_2cell.append(pred_2cell)
                    preds_hierarchical.append(pred_hierarchical)
                    
                    # Store anomalous cells info for localization
                    if pred_hierarchical == 1:
                        anomalous_samples[sample_idx] = anomalous_cells
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        component_errors = np.concatenate(component_errors, axis=0)
        edge_errors = np.concatenate(edge_errors, axis=0)          # Added for 1-cells
        cell2_errors = np.concatenate(cell2_errors, axis=0)        # Added for 2-cells
        
        if len(component_errors) == 0:  # Check if list is empty before concatenation
             print(f"Warning: No samples processed during {eval_mode} evaluation.")
             return {'precision': 0, 'recall': 0, 'f1': 0, 'component_errors': [], 'predictions': [], 'labels': [], 'group_errors': None}
             
        # Compute metrics
        tp = np.sum((all_preds == 1) & (all_labels == 1))
        fp = np.sum((all_preds == 1) & (all_labels == 0))
        fn = np.sum((all_preds == 0) & (all_labels == 1))
        tn = np.sum((all_preds == 0) & (all_labels == 0)) # Calculate true negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 # Calculate specificity

        # Report additional metrics on component thresholds
        if self.use_component_thresholds and self.component_thresholds:
             # *** Restored Code Block ***
            anomalous_components = np.zeros(component_errors.shape[1])
            # Ensure component_errors exists and is not empty before iterating
            if component_errors.size > 0: 
                for i in range(component_errors.shape[0]):
                    for j in range(component_errors.shape[1]):
                        # Check if index j is valid for component_thresholds
                        if j < len(self.component_thresholds) and component_errors[i, j] > self.component_thresholds[j]:
                            anomalous_components[j] += 1
                
                print("\nComponent-level anomaly distribution:")
                # Check if anomalous_components has any counts before sorting
                if np.any(anomalous_components > 0):
                     top_components = np.argsort(anomalous_components)[::-1][:5] # Show top 5
                     for i in top_components:
                         print(f"Component {i}: {anomalous_components[i]} anomalies")
                else:
                     print("No components exceeded their individual thresholds.")
            else:
                 print("Warning: Cannot report component-level distribution as component_errors is empty.")
             # *** End Restored Code Block ***

        print(f"{eval_mode} results with {error_type} errors:")
        print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}, Total: {len(all_labels)}")
        
        # Report temporal consistency effect
        if self.temporal_consistency > 1:
             # *** Restored Code Block ***
            # Count how many anomalies were filtered by temporal consistency
            # Ensure anomaly_counts is not empty before processing
            if anomaly_counts:
                 raw_anomalies = np.sum(np.array(list(anomaly_counts.values())) > 0)
                 filtered_anomalies = np.sum(all_preds)
                 print(f"Temporal consistency filtered out {raw_anomalies - filtered_anomalies} of {raw_anomalies} potential anomalies ({filtered_anomalies} remain)")
            else:
                 print("Temporal consistency: No anomaly counts recorded.")
             # *** End Restored Code Block ***

        print(f"--- End {eval_mode} Evaluation ---")

        # Convert prediction lists to numpy arrays
        preds_0cell = np.array(preds_0cell)
        preds_1cell = np.array(preds_1cell)
        preds_2cell = np.array(preds_2cell)
        preds_hierarchical = np.array(preds_hierarchical)
        
        # Compute metrics for each approach
        metrics = {}
        for approach, preds in [
            ('0-Cell Based', preds_0cell),
            ('1-Cell Based', preds_1cell),
            ('2-Cell Based', preds_2cell),
            ('Hierarchical Approach', preds_hierarchical)
        ]:
            tp = np.sum((preds == 1) & (all_labels == 1))
            fp = np.sum((preds == 1) & (all_labels == 0))
            fn = np.sum((preds == 0) & (all_labels == 1))
            tn = np.sum((preds == 0) & (all_labels == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[approach] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            }
            
            print(f"\n{approach} results:")
            print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        
        # For regular reporting, use the hierarchical approach results
        f1 = metrics['Hierarchical Approach']['f1']
        precision = metrics['Hierarchical Approach']['precision']
        recall = metrics['Hierarchical Approach']['recall']
        
        # Write hierarchical localization results to file
        with open('hierarchical_localization_log.txt', 'w') as f:
            f.write("===== Sample-Level Metrics =====\n")
            for approach, m in metrics.items():
                f.write(f"{approach}:\n")
                f.write(f"F1: {m['f1']:.4f}, Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}\n")
                f.write(f"TP: {m['tp']}, FP: {m['fp']}, FN: {m['fn']}, TN: {m['tn']}\n\n")
            
            f.write("===== Localization Details =====\n")
            for sample_idx, cells in anomalous_samples.items():
                f.write(f"Sample {sample_idx} (True Label: {all_labels[sample_idx]}):\n")
                
                # Write 2-cell anomalies
                f.write("- Anomalous 2-cells:\n")
                if cells['cell2']:
                    for cell in cells['cell2']:
                        cell_name = f"PLC_{cell['id']+1}"  # PLC naming convention
                        f.write(f"  - {cell_name}: error={cell['error']:.4f} (threshold={cell['threshold']:.4f})\n")
                else:
                    f.write("  None\n")
                
                # Write 1-cell anomalies
                f.write("- Anomalous 1-cells:\n")
                if cells['cell1']:
                    for cell in cells['cell1']:
                        cell_name = f"edge_{cell['id']}"
                        f.write(f"  - {cell_name}: error={cell['error']:.4f} (threshold={cell['threshold']:.4f})\n")
                else:
                    f.write("  None\n")
                
                # Write 0-cell anomalies
                f.write("- Anomalous 0-cells:\n")
                if cells['cell0']:
                    for cell in cells['cell0']:
                        cell_name = f"component_{cell['id']}"  # Default generic name
                        if component_names is not None and cell['id'] < len(component_names):
                            cell_name = component_names[cell['id']]
                        f.write(f"  - {cell_name}: error={cell['error']:.4f} (threshold={cell['threshold']:.4f})\n")
                else:
                    f.write("  None\n")
                
                f.write("\n")
            
            f.write(f"Total anomalous samples: {len(anomalous_samples)}\n")
        
        print(f"\nHierarchical localization results written to hierarchical_localization_log.txt")

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'component_errors': component_errors,
            'edge_errors': edge_errors,
            'cell2_errors': cell2_errors,
            'predictions': all_preds,
            'labels': all_labels,
            'preds_0cell': preds_0cell,
            'preds_1cell': preds_1cell,
            'preds_2cell': preds_2cell,
            'preds_hierarchical': preds_hierarchical,
            'metrics': metrics,
            'anomalous_samples': anomalous_samples
        }

    def train(self, num_epochs=10, test_interval=1, checkpoint_dir="model_checkpoints", 
              early_stopping=True, patience=3, min_delta=1e-4):
        """
        Train the model for multiple epochs, calibrating thresholds on validation data
        but only evaluating performance on test data.
        
        Parameters
        ----------
        num_epochs : int, default=10
            Maximum number of epochs to train
        test_interval : int, default=1
            Frequency of evaluation on test set
        checkpoint_dir : str, default="model_checkpoints"
            Directory to save model checkpoints
        early_stopping : bool, default=True
            Whether to use early stopping based on test F1 score
        patience : int, default=3
            Number of epochs to wait for improvement before stopping
        min_delta : float, default=1e-4
            Minimum change in test F1 to qualify as improvement
        """
        print(f"\n--- Starting Training for {num_epochs} epochs ---")
        print(f"Calibrating thresholds on validation set every {test_interval} epochs.")
        print(f"Performance metrics will only be calculated on test data.")
        
        if early_stopping:
            print(f"Using early stopping with patience={patience}, min_delta={min_delta}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        best_test_f1 = -1.0  # Initialize with -1 to ensure first F1 score is better
        best_epoch = 0
        epochs_without_improvement = 0  # Counter for early stopping
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch
            train_loss = self.train_epoch()
            print(f"Epoch {epoch+1} Train Loss: {train_loss:.6f}")
            
            # Update learning rate based on loss
            self.scheduler.step(train_loss)
            
            # Evaluate at specified intervals
            if (epoch + 1) % test_interval == 0 or epoch == num_epochs - 1:
                # Calibrate thresholds using validation data
                print("\n--- Calibrating thresholds on validation data ---")
                self.calibrate_threshold()
                
                # Evaluate on TEST data using the calibrated thresholds
                print("\n--- Evaluating on TEST data ---")
                test_results = self.evaluate(dataloader=self.test_dataloader, eval_mode="Test")
                
                # Check if evaluation returned results
                if test_results and 'f1' in test_results:
                    test_f1 = test_results['f1']
                    test_precision = test_results['precision']
                    test_recall = test_results['recall']
                    
                    print(f"Epoch {epoch+1} Test - F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
                    
                    # Save model checkpoint (always save the latest)
                    checkpoint_path = os.path.join(checkpoint_dir, f"model_ckpt_ep_{epoch+1}.pt")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(),
                        'loss': train_loss,
                        'test_f1': test_f1,
                        'anomaly_threshold': self.anomaly_threshold,
                        'component_thresholds': self.component_thresholds,
                        'cell2_threshold': self.cell2_threshold,
                        'cell2_component_thresholds': self.cell2_component_thresholds
                    }, checkpoint_path)
                    print(f"Saved model checkpoint to {checkpoint_path}")

                    # Check for improvement
                    improved = False
                    if test_f1 - best_test_f1 > min_delta:
                        improved = True
                        best_test_f1 = test_f1
                        best_epoch = epoch + 1
                        epochs_without_improvement = 0
                        print(f"*** New best test F1: {best_test_f1:.4f} at epoch {best_epoch} ***")
                        
                        # Save best model based on test performance
                        best_model_path = os.path.join(checkpoint_dir, "best_test_ep.pt")
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.opt.state_dict(),
                            'loss': train_loss,
                            'test_f1': test_f1,
                            'test_precision': test_precision,
                            'test_recall': test_recall,
                            'anomaly_threshold': self.anomaly_threshold,
                            'component_thresholds': self.component_thresholds,
                            'cell2_threshold': self.cell2_threshold,
                            'cell2_component_thresholds': self.cell2_component_thresholds
                        }, best_model_path)
                        print(f"Saved best test model to {best_model_path}")
                        
                        # Store the last evaluation results for potential localization
                        self.last_evaluation_results = test_results
                    else:
                        # If no improvement, increment counter
                        epochs_without_improvement += 1
                        if early_stopping and epochs_without_improvement >= patience:
                            print(f"\n--- Early stopping triggered: No improvement for {patience} epochs ---")
                            print(f"Best test F1: {best_test_f1:.4f} achieved at epoch {best_epoch}")
                            break
                        else:
                            print(f"No improvement. Best F1: {best_test_f1:.4f} at epoch {best_epoch}. "
                                  f"Epochs without improvement: {epochs_without_improvement}/{patience}")
                else:
                    print(f"Epoch {epoch+1} - Skipping test F1 update due to evaluation error.")

            epoch_end_time = time.time()
            print(f"Epoch {epoch+1} duration: {epoch_end_time - epoch_start_time:.2f} seconds")
        
        print(f"\n--- Training Completed ---")
        print(f"Best test F1: {best_test_f1:.4f} achieved at epoch {best_epoch}")
        print(f"Best test model saved at: {os.path.join(checkpoint_dir, 'best_test_ep.pt')}")
        
        return self.last_evaluation_results  # Return the evaluation results from the best model

    # Add localize_anomalies method (ensure it uses the correct thresholds)
    def localize_anomalies(self, component_names):
        """
        Localize anomalies hierarchically from 2-cells to 1-cells to 0-cells.
        """
        print("\n--- Hierarchical Anomaly Localization ---")
        
        # Check threshold availability
        if self.anomaly_threshold is None or self.cell1_threshold is None or self.cell2_threshold is None:
            print("Error: Thresholds not set. Cannot localize anomalies.")
            return {}
        
        # Get results from last evaluation
        if not hasattr(self, 'last_evaluation_results') or not self.last_evaluation_results:
            print("Running evaluation on test set again to get errors for localization...")
            self.last_evaluation_results = self.evaluate(dataloader=self.test_dataloader, eval_mode="Localization")
        
        preds = self.last_evaluation_results['predictions']
        labels = self.last_evaluation_results['labels']
        component_errors = self.last_evaluation_results['component_errors']
        edge_errors = self.last_evaluation_results['edge_errors']
        cell2_errors = self.last_evaluation_results['cell2_errors']
        
        # Get the relationship matrices from the first sample in the dataset
        sample = next(iter(self.test_dataloader))
        if isinstance(sample, tuple) and len(sample) >= 8:
            _, _, _, _, _, _, b1, b2 = sample[:8]
            
            # Convert to numpy for easier processing
            if isinstance(b1, torch.Tensor):
                b1_matrix = b1.cpu().numpy()
                if b1_matrix.ndim > 2:  # Remove batch dimension if present
                    b1_matrix = b1_matrix[0]
            
            if isinstance(b2, torch.Tensor):
                b2_matrix = b2.cpu().numpy()
                if b2_matrix.ndim > 2:  # Remove batch dimension if present
                    b2_matrix = b2_matrix[0]
            
            # Get relationship mappings
            cell2_to_cell1 = {}  # Maps 2-cell index to list of 1-cell indices
            cell1_to_cell0 = {}  # Maps 1-cell index to list of 0-cell indices
            
            # Build 2-cell to 1-cell mapping
            for cell2_idx in range(b2_matrix.shape[1]):
                cell2_to_cell1[cell2_idx] = np.where(b2_matrix[:, cell2_idx] > 0)[0].tolist()
            
            # Build 1-cell to 0-cell mapping
            for cell1_idx in range(b1_matrix.shape[1]):
                cell1_to_cell0[cell1_idx] = np.where(b1_matrix[:, cell1_idx] > 0)[0].tolist()
        else:
            print("Warning: Could not extract relationship matrices from dataset")
            cell2_to_cell1 = {}
            cell1_to_cell0 = {}
        
        # Create list to store all localization results
        hierarchical_results = []
        
        # Process each sample
        for i in range(len(preds)):
            if preds[i] == 1 or labels[i] == 1:  # Focus on predicted or actual anomalies
                sample_result = {
                    'sample_idx': i,
                    'predicted': preds[i],
                    'actual': labels[i],
                    'cell2_anomalies': [],
                    'cell1_anomalies': [],
                    'cell0_anomalies': []
                }
                
                # Check 2-cells
                for cell2_idx in range(cell2_errors.shape[1]):
                    cell2_error = cell2_errors[i, cell2_idx]
                    is_anomalous = cell2_error > self.cell2_threshold
                    
                    cell2_info = {
                        'id': f"PLC_{cell2_idx+1}",  # Use PLC_N naming convention
                        'error': float(cell2_error),
                        'threshold': float(self.cell2_threshold),
                        'is_anomalous': bool(is_anomalous),
                        'related_cell1': []
                    }
                    
                    # If 2-cell is anomalous, check its 1-cells
                    if is_anomalous:
                        sample_result['cell2_anomalies'].append(cell2_info)
                        
                        # Get related 1-cells
                        related_cell1 = cell2_to_cell1.get(cell2_idx, [])
                        for cell1_idx in related_cell1:
                            cell1_error = edge_errors[i, cell1_idx]
                            cell1_is_anomalous = cell1_error > self.cell1_threshold
                            
                            cell1_info = {
                                'id': f"edge_{cell1_idx}",
                                'error': float(cell1_error),
                                'threshold': float(self.cell1_threshold),
                                'is_anomalous': bool(cell1_is_anomalous),
                                'related_cell0': []
                            }
                            
                            # If 1-cell is anomalous, check its 0-cells
                            if cell1_is_anomalous:
                                cell2_info['related_cell1'].append(cell1_info)
                                sample_result['cell1_anomalies'].append(cell1_info)
                                
                                # Get related 0-cells
                                related_cell0 = cell1_to_cell0.get(cell1_idx, [])
                                for cell0_idx in related_cell0:
                                    if cell0_idx < len(component_names):
                                        cell0_error = component_errors[i, cell0_idx]
                                        cell0_is_anomalous = cell0_error > self.anomaly_threshold
                                        
                                        cell0_info = {
                                            'id': component_names[cell0_idx],
                                            'error': float(cell0_error),
                                            'threshold': float(self.anomaly_threshold),
                                            'is_anomalous': bool(cell0_is_anomalous)
                                        }
                                        
                                        if cell0_is_anomalous:
                                            cell1_info['related_cell0'].append(cell0_info)
                                            sample_result['cell0_anomalies'].append(cell0_info)
            
                # Only include samples with anomalies at any level
                if (sample_result['cell2_anomalies'] or sample_result['cell1_anomalies'] 
                        or sample_result['cell0_anomalies']):
                    hierarchical_results.append(sample_result)
        
        # Write results to file
        with open('hierarchical_localization_log.txt', 'w') as f:
            f.write("=== Hierarchical Anomaly Localization Results ===\n\n")
            
            for result in hierarchical_results:
                f.write(f"Sample {result['sample_idx']} (Predicted: {result['predicted']}, Actual: {result['actual']}):\n")
                
                # Write 2-cell results
                for cell2 in result['cell2_anomalies']:
                    status = "ANOMALOUS" if cell2['is_anomalous'] else "NORMAL"
                    f.write(f"- 2-cell {cell2['id']}: error={cell2['error']:.4f} (threshold={cell2['threshold']:.4f}) - {status}\n")
                    
                    # Write related 1-cell results
                    for cell1 in cell2['related_cell1']:
                        status = "ANOMALOUS" if cell1['is_anomalous'] else "NORMAL"
                        f.write(f"  - 1-cell {cell1['id']}: error={cell1['error']:.4f} (threshold={cell1['threshold']:.4f}) - {status}\n")
                        
                        # Write related 0-cell results
                        for cell0 in cell1['related_cell0']:
                            status = "ANOMALOUS" if cell0['is_anomalous'] else "NORMAL"
                            f.write(f"    - 0-cell {cell0['id']}: error={cell0['error']:.4f} (threshold={cell0['threshold']:.4f}) - {status}\n")
                
                f.write("\n")
            
            f.write(f"Total anomalous samples: {len(hierarchical_results)}\n")
        
        print(f"Hierarchical localization complete. Results written to hierarchical_localization_log.txt")
        self.hierarchical_results = hierarchical_results
        return hierarchical_results

def load_swat_data(train_path, test_path, sample_rate=1.0, save_test_path=None, validation_split_ratio=0.2, 
                  attack_focused_sampling=False, pre_attack_points=10, attack_start_points=5):
    """
    Load SWAT dataset from CSV files with optional sampling and validation split.

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
    print(f"Loading SWAT data from {train_path} and {test_path}...")
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
            attack_indices, _ = get_attack_indices("SWAT")
            
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
        # Ensure reproducibility
        #np.random.seed(42) 

        # Shuffle indices
        #shuffled_indices = np.random.permutation(initial_train_data.index)
        
        # Calculate split point
        #split_idx = int(len(initial_train_data) * (1 - validation_split_ratio))
        
        # Get train and validation indices
        #train_indices = initial_train_data.index[:split_idx]
        #validation_indices = shuffled_indices[split_idx:]
        # Calculate split point based on the *current* length (after potential sampling)
        split_idx = int(len(initial_train_data) * (1 - validation_split_ratio))

        # --- MODIFICATION START ---
        # Get indices *without* shuffling to preserve order from linspace sampling
        # This ensures validation set maintains the sequence if sample_rate < 1.0
        all_indices = initial_train_data.index.to_numpy() # Use current index after sampling
        train_indices = all_indices[:split_idx]
        validation_indices = all_indices[split_idx:]
        print(f"Splitting train data sequentially (no shuffle) for validation.") # Add print statement
        # --- MODIFICATION END ---
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

    # Convert 'Normal/Attack' to boolean/int for consistency
    train_data['Normal/Attack'] = train_data['Normal/Attack'].map(lambda x: 0 if (x == 'False' or x == 0 or x == 0.0 or x == 'Normal') else 1)
    if not validation_data.empty:
        validation_data['Normal/Attack'] = validation_data['Normal/Attack'].map(lambda x: 0 if (x == 'False' or x == 0 or x == 0.0 or x == 'Normal') else 1)
    # Ensure test data labels are also 0 or 1
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

def analyze_attack_detection(trainer, anomaly_map, component_names):
    """
    Analyze which attacks are being detected and how accurately.
    Maps anomalies back to specific attack scenarios from attack_utils.
    """
    # Get attack details
    attacks_sds = get_attack_sds("SWAT")
    
    # Create mapping of component name to attack ID
    component_to_attack = {}
    for attack_id, attack in enumerate(attacks_sds):
        _, component, _, _, _ = attack
        if component not in component_to_attack:
            component_to_attack[component] = []
        component_to_attack[component].append(attack_id)
    
    # Count attacks detected by target component
    attack_detection = {}
    
    # Fix: Properly iterate through the keys that actually exist in anomaly_map
    for idx in anomaly_map.keys():  # Use the actual keys from the dictionary
        details = anomaly_map[idx]
        if details['y_true'] == 1:  # This is an actual attack
            detected = (details['y_pred'] == 1)
            
            # Check which components were flagged
            flagged_components = [c['component'] for c in details['components'] 
                                if c['is_above_threshold']]
            
            # Check which attack targets match our flagged components
            for component in flagged_components:
                if component in component_to_attack:
                    for attack_id in component_to_attack[component]:
                        if attack_id not in attack_detection:
                            attack_detection[attack_id] = {'correct': 0, 'total': 0}
                        if detected:
                            attack_detection[attack_id]['correct'] += 1
                        attack_detection[attack_id]['total'] += 1
    
    # Rest of the function remains the same...

def main():
    """Modified main function to run SWAT experiment in reconstruction mode."""
    # ========= Reproducibility ==========
    seed = 42 # You can choose any fixed integer
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print(f"Set random seed to {seed} for reproducibility.")
    # ==================================
    
    print("Starting SWAT Anomaly Detection in Reconstruction Mode...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define paths
    data_dir = "data/SWAT"
    train_path = os.path.join(data_dir, "SWATv0_train.csv")
    test_path = os.path.join(data_dir, "SWATv0_test.csv")
    
    # Set up run folder structure
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join("runs", f"run_{timestamp}")
    checkpoint_dir = os.path.join(run_dir, "model_checkpoints")
    data_save_dir = os.path.join(run_dir, "data") 
    results_dir = os.path.join(run_dir, "results")

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(data_save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Define paths for saving data
    test_data_save_path = os.path.join(data_save_dir, "test_data.csv")
    validation_data_save_path = os.path.join(data_save_dir, "validation_data.csv")

    # ========= KEY PARAMETERS ==========
    sample_rate = 0.001 # Increased sample rate for more data
    validation_split_ratio = 0.2 
    
    # Anomaly detection parameters
    use_geometric_mean = False  # Use regular mean instead of geometric mean
    threshold_method = "percentile" 
    threshold_percentile = 99.00  
    use_component_thresholds = False  # Use global threshold instead of component-specific
    temporal_consistency = 1
    # ==================================

    # Load data with validation split and optional attack-focused sampling
    train_data, validation_data, test_data = load_swat_data(
        train_path, 
        test_path, 
        sample_rate=sample_rate,
        save_test_path=test_data_save_path,
        validation_split_ratio=validation_split_ratio,
        attack_focused_sampling=False
    )

    # Save validation data if exists
    if not validation_data.empty:
        validation_data.to_csv(validation_data_save_path, index=False)

    # Initialize SWAT complex
    print("Building SWAT topology...")
    swat_complex = SWATComplex()

    # Get component names
    component_names = [col for col in train_data.columns if col not in ['Timestamp', 'Normal/Attack']]

    # Create datasets with reconstruction mode (temporal_mode=False)
    print("Creating datasets (train, validation, test) in reconstruction mode...")
    train_dataset = SWaTDataset(train_data, swat_complex, temporal_mode=False)
    validation_dataset = SWaTDataset(validation_data, swat_complex, temporal_mode=False) if not validation_data.empty else None
    test_dataset = SWaTDataset(test_data, swat_complex, temporal_mode=False)

    # Create dataloaders
    batch_size = 1
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False) if validation_dataset else None
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Check if validation dataloader exists
    if validation_dataloader is None:
         print("Error: Validation dataloader could not be created. Check validation_split_ratio.")
         return

    # Define model hyperparameters
    #lstm_hidden_dim = 64
    #in_channels = [lstm_hidden_dim, lstm_hidden_dim, lstm_hidden_dim]
    in_channels = [3, 3, 3]
    intermediate_channels = [32, 32, 32]
    final_channels = [64, 64, 64]
    channels_per_layer = [[in_channels, intermediate_channels, final_channels]]

    # Create model in reconstruction mode (temporal_mode=False)
    print("Creating model in reconstruction mode...")
    model = AnomalyCCANN(
        channels_per_layer, 
        original_feature_dim=3,
        temporal_mode=False  # Set to reconstruction mode
    )

    # Create trainer with validation dataloader
    print("Creating trainer with validation set for thresholding...")
    trainer = AnomalyTrainer(
        model,
        train_dataloader,
        validation_dataloader,
        test_dataloader,
        learning_rate=0.005,
        device=device,
        threshold_percentile=threshold_percentile,
        use_geometric_mean=use_geometric_mean,
        epsilon=1e-6,
        threshold_method=threshold_method,
        sd_multiplier=2.5,
        use_component_thresholds=use_component_thresholds, 
        temporal_consistency=temporal_consistency,
        weight_decay=1e-5,
        grad_clip_value=1.0
    )
    
    # Train model with early stopping
    final_test_results = trainer.train(
        num_epochs=5,  # Reduced number of epochs for reconstruction mode
        test_interval=1, 
        checkpoint_dir=checkpoint_dir,
        early_stopping=True,
        patience=3,
        min_delta=1e-4
    )
    
    # Localize anomalies using the final model state and thresholds
    if final_test_results:
         trainer.last_evaluation_results = final_test_results 
         anomaly_map = trainer.localize_anomalies(component_names)

         print(f"\nFinal Test Set evaluation - F1: {final_test_results['f1']:.4f}, Precision: {final_test_results['precision']:.4f}, Recall: {final_test_results['recall']:.4f}")
         print(f"Detected {sum(final_test_results['predictions'])} anomalous samples out of {len(final_test_results['labels'])} total test samples")
         
         # Run attack analysis if localization was successful
         if anomaly_map:
             # In the main() function, wrap the analyze_attack_detection call in a try-except block:
             try:
                 attack_detection = analyze_attack_detection(trainer, anomaly_map, component_names)
                 # Save attack detection results
                 if attack_detection:
                     np.savez(os.path.join(results_dir, "attack_detection.npz"), attack_detection=attack_detection)
             except AttributeError as e:
                 print(f"Skipping attack detection analysis")
                 
         else:
             print("Skipping attack analysis due to localization issues.")
    else:
         print("Skipping final analysis as training/evaluation did not complete successfully.")

    print(f"\nReconstruction mode experiment completed!")
    print(f"All outputs have been saved under the directory: {run_dir}")

if __name__ == "__main__":
    main()
