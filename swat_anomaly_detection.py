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
from utils.attack_utils import get_attack_indices, is_actuator
import time

class SWaTDataset(Dataset):
    """Dataset class for SWAT anomaly detection."""
    def __init__(self, data, swat_complex, feature_dim=3):
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
        """
        self.data = data
        self.complex = swat_complex.get_complex()
        self.feature_dim = feature_dim

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

        print(f"Initialized SWaTDataset with {len(self.data)} samples")
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
        B = self.complex.incidence_matrix(1, 2)
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
        return len(self.data)

    def __getitem__(self, idx):
        """Return a sample from the dataset."""
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

class AnomalyCCANN(nn.Module):
    """
    Anomaly detection model using Combinatorial Complex Attention Neural Network.
    Uses an autoencoder approach: encodes features using HMC, then decodes
    to reconstruct the original 0-cell features.
    """
    def __init__(self, channels_per_layer, original_feature_dim):
        """
        Initialize the anomaly detection model.

        Parameters
        ----------
        channels_per_layer : list
            List of layer configurations for the HMC encoder
        original_feature_dim : int
            Original feature dimension for 0-cells
        """
        super().__init__()
        self.encoder = HMC(channels_per_layer)

        # Get the output dimension of the last layer for 0-cells
        final_dim = channels_per_layer[-1][2][0]

        # Decoder to reconstruct 0-cell features
        self.decoder = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.ReLU(),
            nn.Linear(64, original_feature_dim)
        )

        print(f"Initialized AnomalyCCANN with encoder output dim {final_dim} and decoder output dim {original_feature_dim}")

    def forward(self, x_0, x_1, x_2, a0, a1, coa2, b1, b2):
        """
        Forward pass through the model.
        
        Parameters
        ----------
        x_0 : torch.Tensor
            Features for 0-cells
        x_1 : torch.Tensor
            Features for 1-cells
        x_2 : torch.Tensor
            Features for 2-cells
        a0 : torch.sparse.Tensor
            Adjacency matrix for 0-cells
        a1 : torch.sparse.Tensor
            Adjacency matrix for 1-cells
        coa2 : torch.sparse.Tensor
            Coadjacency matrix for 2-cells
        b1 : torch.sparse.Tensor
            Incidence matrix from 0-cells to 1-cells
        b2 : torch.sparse.Tensor
            Incidence matrix from 1-cells to 2-cells

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
        
        # For sparse tensors, we need to handle them differently
        # First, check if they have >2 dimensions (i.e., have a batch dimension)
        if a0.dim() > 2:
            # No need to print this for every sample
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
            x_0_enc, _, _ = self.encoder(x_0_no_batch, x_1_no_batch, x_2_no_batch, 
                                         a0_no_batch, a1_no_batch, coa2_no_batch, 
                                         b1_no_batch, b2_no_batch)
        except Exception as e:
            print(f"ERROR in encoder: {e}")
            raise
        
        # Decode to reconstruct 0-cell features
        x_0_recon_no_batch = self.decoder(x_0_enc)
        
        # Add back the batch dimension
        x_0_recon = x_0_recon_no_batch.unsqueeze(0)
        
        return x_0_recon

class AnomalyTrainer:
    """
    Trainer for the anomaly detection model.
    Trains the model on benign data, then detects anomalies
    based on reconstruction error.
    """
    def __init__(self, model, train_dataloader, test_dataloader, learning_rate, device, threshold_percentile=95):
        """
        Initialize the trainer.

        Parameters
        ----------
        model : AnomalyCCANN
            The model to train
        train_dataloader : torch.utils.data.DataLoader
            DataLoader for training data (benign only)
        test_dataloader : torch.utils.data.DataLoader
            DataLoader for testing data (may include anomalies)
        learning_rate : float
            Learning rate for the optimizer
        device : torch.device
            Device to use for training
        threshold_percentile : int, default=95
            Percentile of benign reconstruction errors to use as anomaly threshold
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.threshold_percentile = threshold_percentile
        self.crit = nn.MSELoss(reduction='none')  # Keep per-element losses for analysis
        self.opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.anomaly_threshold = None
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='min', factor=0.5, patience=3, verbose=True
        )

        print(f"Initialized AnomalyTrainer with device {device} and threshold percentile {threshold_percentile}")

    def to_device(self, x):
        """
        Move all tensors to the device.

        Parameters
        ----------
        x : list of torch.Tensor
            List of tensors to move

        Returns
        -------
        list of torch.Tensor
            List of tensors moved to the device
        """
        return [el.float().to(self.device) for el in x]

    def train(self, num_epochs=50, test_interval=10):
        """
        Train the model.

        Parameters
        ----------
        num_epochs : int, default=50
            Number of epochs to train
        test_interval : int, default=10
            Interval between testing epochs
        """
        print(f"Starting training for {num_epochs} epochs...")
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            print(f"Epoch: {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}", flush=True)
            
            # Save model checkpoint after each epoch
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.opt.state_dict(),
                'loss': train_loss,
                'threshold': self.anomaly_threshold
            }
            torch.save(checkpoint, f'model_checkpoint_epoch_{epoch+1}.pt')
            print(f"Saved checkpoint for epoch {epoch+1}")
            
            # Track best model
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(checkpoint, 'best_model.pt')
                print(f"New best model (loss: {best_loss:.6f})")

            if (epoch + 1) % test_interval == 0:
                # Evaluate on training set to determine threshold
                self.calibrate_threshold()

                # Evaluate on test set to detect anomalies
                test_results = self.evaluate()
                print(f"Test Results - F1: {test_results['f1']:.4f}, Precision: {test_results['precision']:.4f}, Recall: {test_results['recall']:.4f}")

            self.scheduler.step(train_loss)

        # Print training summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Trained for {num_epochs} epochs")
        print(f"Best loss: {best_loss:.6f}")
        print(f"Final loss: {train_loss:.6f}")
        print(f"Final threshold: {self.anomaly_threshold:.6f}")
        print("="*50)

    def train_epoch(self):
        """
        Train the model for one epoch.

        Returns
        -------
        float
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Get total number of samples to track progress
        total_samples = len(self.train_dataloader)
        print(f"Training on {total_samples} samples...")
        
        # Progress tracking
        log_interval = max(1, total_samples // 10)  # Show progress ~10 times per epoch
        start_time = time.time()

        for i, sample in enumerate(self.train_dataloader):
            # Move all tensors to device
            x_0, x_1, x_2, a0, a1, coa2, b1, b2, _ = self.to_device(sample)

            # Zero gradients
            self.opt.zero_grad()

            # Forward pass
            x_0_recon = self.model(x_0, x_1, x_2, a0, a1, coa2, b1, b2)

            # Compute loss (MSE)
            loss = self.crit(x_0_recon, x_0).mean()

            # Backward pass and optimize
            loss.backward()
            self.opt.step()

            total_loss += loss.item()
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

        return total_loss / num_batches

    def calibrate_threshold(self):
        """
        Calibrate the anomaly detection threshold based on benign data.
        Uses the specified percentile of reconstruction errors on benign data.
        """
        print(f"Calibrating anomaly threshold at {self.threshold_percentile}th percentile...")
        self.model.eval()
        all_errors = []

        with torch.no_grad():
            for sample in self.train_dataloader:
                # Move all tensors to device
                x_0, x_1, x_2, a0, a1, coa2, b1, b2, _ = self.to_device(sample)

                # Forward pass
                x_0_recon = self.model(x_0, x_1, x_2, a0, a1, coa2, b1, b2)

                # Compute reconstruction error for each sample and component
                error = self.crit(x_0_recon, x_0).mean(dim=2)  # [batch_size, num_components]
                all_errors.append(error.cpu().numpy())

        # Concatenate all errors
        all_errors = np.concatenate(all_errors, axis=0)

        # Compute threshold as the specified percentile
        self.anomaly_threshold = np.percentile(all_errors, self.threshold_percentile)
        print(f"Calibrated anomaly threshold: {self.anomaly_threshold:.6f}")

    def evaluate(self):
        """
        Evaluate the model on test data.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics
        """
        if self.anomaly_threshold is None:
            print("Warning: Threshold not calibrated. Calibrating now...")
            self.calibrate_threshold()

        print("Evaluating model on test data...")
        self.model.eval()
        all_preds = []
        all_labels = []
        component_errors = []

        with torch.no_grad():
            for sample in self.test_dataloader:
                # Move all tensors to device
                x_0, x_1, x_2, a0, a1, coa2, b1, b2, y = self.to_device(sample)

                # Forward pass
                x_0_recon = self.model(x_0, x_1, x_2, a0, a1, coa2, b1, b2)

                # Compute reconstruction error for each sample and component
                error = self.crit(x_0_recon, x_0).mean(dim=2)  # [batch_size, num_components]

                # Get sample-level error (max error across components)
                sample_error = error.max(dim=1)[0]

                # Predict anomalies
                pred = (sample_error > self.anomaly_threshold).float()

                all_preds.append(pred.cpu().numpy())
                all_labels.append(y.cpu().numpy())
                component_errors.append(error.cpu().numpy())

        # Concatenate all predictions and labels
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        component_errors = np.concatenate(component_errors, axis=0)

        # Compute metrics
        tp = np.sum((all_preds == 1) & (all_labels == 1))
        fp = np.sum((all_preds == 1) & (all_labels == 0))
        fn = np.sum((all_preds == 0) & (all_labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Identify most anomalous components
        anomaly_indices = np.where(all_labels == 1)[0]
        if len(anomaly_indices) > 0:
            attack_errors = component_errors[anomaly_indices]
            avg_component_errors = np.mean(attack_errors, axis=0)
            top_components = np.argsort(avg_component_errors)[::-1][:5]
            print(f"Top 5 anomalous components: {top_components}")

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'component_errors': component_errors,
            'predictions': all_preds,
            'labels': all_labels
        }

def localize_anomalies(self, component_names):
    """Localize anomalies to specific components with better reporting."""
    if self.anomaly_threshold is None:
        print("Warning: Threshold not calibrated. Calibrating now...")
        self.calibrate_threshold()

    print("Localizing anomalies with improved component ranking...")
    self.model.eval()
    anomaly_map = {}

    with torch.no_grad():
        for batch_idx, sample in enumerate(self.test_dataloader):
            # Move all tensors to device
            x_0, x_1, x_2, a0, a1, coa2, b1, b2, y = self.to_device(sample)

            # Forward pass
            x_0_recon = self.model(x_0, x_1, x_2, a0, a1, coa2, b1, b2)

            # Compute reconstruction error for each component
            error = self.crit(x_0_recon, x_0).mean(dim=2)  # [batch_size, num_components]
            
            # Get sample-level error (max error across components)
            sample_error = error.max(dim=1)[0]
            
            # Predict if sample is anomalous
            is_anomalous = (sample_error > self.anomaly_threshold).item()
            
            # Process each sample
            for i in range(len(y)):
                sample_idx = batch_idx  # Since batch_size is 1
                
                # If it's anomalous (either predicted or actual)
                if is_anomalous or y[i] == 1:
                    # Get component errors
                    component_errors = error[i].cpu().numpy()
                    
                    # Sort components by error (highest first)
                    sorted_indices = np.argsort(component_errors)[::-1]
                    
                    # Get top 5 or all components above threshold
                    anomalous_indices = sorted_indices[:5]  # Top 5 regardless of threshold
                    
                    # Map indices to component names and errors
                    anomalous_components = []
                    for idx in anomalous_indices:
                        anomalous_components.append({
                            'component': component_names[idx],
                            'error': float(component_errors[idx]),
                            'is_above_threshold': component_errors[idx] > self.anomaly_threshold
                        })
                    
                    # Store in map
                    anomaly_map[sample_idx] = {
                        'y_true': y[i].item(),
                        'y_pred': int(is_anomalous),
                        'max_error': sample_error.item(),
                        'components': anomalous_components
                    }

    # Print summary of component frequency in top errors
    component_count = {}
    for details in anomaly_map.values():
        for comp_info in details['components']:
            if comp_info['is_above_threshold']:
                comp_name = comp_info['component']
                component_count[comp_name] = component_count.get(comp_name, 0) + 1
    
    print("\nTop components flagged as anomalous:")
    for comp, count in sorted(component_count.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {comp}: {count} times")

    return anomaly_map

def load_swat_data(train_path, test_path, sample_rate=1.0):
    """
    Load SWAT dataset from CSV files with optional sampling.

    Parameters
    ----------
    train_path : str
        Path to training data CSV
    test_path : str
        Path to testing data CSV
    sample_rate : float, default=1.0
        Fraction of data to sample (0.0-1.0). Use 1.0 for full dataset.

    Returns
    -------
    tuple of pandas.DataFrame
        (train_data, test_data)
    """
    print(f"Loading SWAT data from {train_path} and {test_path}...")
    print(f"Using sample rate: {sample_rate}")

    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Apply sampling if needed
    if sample_rate < 1.0:
        # Calculate sample sizes
        train_sample_size = int(len(train_data) * sample_rate)
        test_sample_size = int(len(test_data) * sample_rate)
        
        # Sample data - using systematic sampling to maintain temporal patterns
        train_indices = np.linspace(0, len(train_data)-1, train_sample_size, dtype=int)
        test_indices = np.linspace(0, len(test_data)-1, test_sample_size, dtype=int)
        
        train_data = train_data.iloc[train_indices].reset_index(drop=True)
        test_data = test_data.iloc[test_indices].reset_index(drop=True)
        
        print(f"Sampled data: train={len(train_data)} rows, test={len(test_data)} rows")
    else:
        print(f"Using full dataset: train={len(train_data)} rows, test={len(test_data)} rows")

# Convert 'Normal/Attack' to boolean for consistency
    train_data['Normal/Attack'] = train_data['Normal/Attack'].astype(str)
    test_data['Normal/Attack'] = test_data['Normal/Attack']
    # Check label distribution and print it for verification
    normal_test = (test_data['Normal/Attack'] == 0.0).sum()
    attack_test = (test_data['Normal/Attack'] == 1.0).sum()
    print(f"Test data contains {normal_test} normal samples and {attack_test} attack samples")
    print(f"Attack percentage: {attack_test/len(test_data)*100:.2f}%")


    return train_data, test_data

def main():
    """Main function to run the SWAT anomaly detection."""
    print("Starting SWAT Anomaly Detection...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define paths
    data_dir = "data/SWAT"
    train_path = os.path.join(data_dir, "SWATv0_train.csv")
    test_path = os.path.join(data_dir, "SWATv0_test.csv")

    # Set sampling rate (use a small value for faster testing)
    #sample_rate = 1.0  # Use 5% of the data for testing
    
    # Load data with sampling
    train_data, test_data = load_swat_data(train_path, test_path)

    # Initialize SWAT complex
    print("Building SWAT topology...")
    swat_complex = SWATComplex()

    # Get component names (excluding Timestamp and Normal/Attack)
    component_names = [col for col in train_data.columns if col not in ['Timestamp', 'Normal/Attack']]

    # Create datasets
    print("Creating datasets...")
    train_dataset = SWaTDataset(train_data, swat_complex)
    test_dataset = SWaTDataset(test_data, swat_complex)

    # Create dataloaders
    batch_size = 1  # Changed from 64 to process one sample at a time
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define model hyperparameters
    feature_dim = 3  # Dimension of cell features

    # Define channels per layer
    in_channels = [feature_dim, feature_dim, feature_dim]
    intermediate_channels = [32, 32, 32]
    final_channels = [64, 64, 64]
    channels_per_layer = [[in_channels, intermediate_channels, final_channels]]

    # Create model
    print("Creating model...")
    model = AnomalyCCANN(channels_per_layer, original_feature_dim=feature_dim)

    # Create trainer
    print("Creating trainer...")
    trainer = AnomalyTrainer(
        model,
        train_dataloader,
        test_dataloader,
        learning_rate=0.001,
        device=device,
        threshold_percentile=99.5  # Higher to reduce false positives
    )

    # Train for more epochs
    trainer.train(num_epochs=10, test_interval=2)

    # Evaluate and localize anomalies
    print("Evaluating model and localizing anomalies...")
    results = trainer.evaluate()
    anomaly_map = trainer.localize_anomalies(component_names)

    print(f"Final evaluation - F1: {results['f1']:.4f}, Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}")
    print(f"Detected {len(anomaly_map)} anomalous samples")

    # Print a few localized anomalies
    for i, (idx, details) in enumerate(anomaly_map.items()):
        if i >= 5:  # Only show first 5
            break
        print(f"Anomaly {idx}: {details['components']}")

    print("SWAT Anomaly Detection completed!")

if __name__ == "__main__":
    main()
