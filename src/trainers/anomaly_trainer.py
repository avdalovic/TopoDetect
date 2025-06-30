import os
import time
import pickle
import numpy as np
import torch
from torch import nn
import wandb

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
                
                # Forward pass - returns tuple of predictions
                if self.model.use_tcn:
                    #TCN returns predictions for all three levels
                    x0_pred, x1_pred, x2_pred = self.model(sample)
                    #Targets for all three levels
                    target_x0 = sample['target_x0']
                    target_x1 = sample['target_x1']
                    target_x2 = sample['target_x2']

                    # Compute loss for each level
                    loss_0 = self.crit(x0_pred, target_x0).mean()
                    loss_1 = self.crit(x1_pred, target_x1).mean()
                    loss_2 = self.crit(x2_pred, target_x2).mean()

                    # Combined loss with equal weights
                    loss = 1/51 * loss_0 + 1/74 * loss_1 + 1/5 * loss_2
                    
                    total_loss += loss_0.item()
                else:
                    # Original LSTM temporal logic unchanged 
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
        # FIX: Handle both temporal and non-temporal modes for debug print
        try:
            sample = next(iter(self.validation_dataloader))
            if self.model.temporal_mode:
                print(f"First validation sample keys: {list(sample.keys())}")
                print(f"First validation sample seq_x0 shape: {sample['seq_x0'].shape}")
            else:
                print(f"First validation sample shape: {sample[0].shape}")
        except Exception as e:
            print(f"Could not get sample shape info: {e}")
        
        # For debugging purposes, make error collection more robust with try/except
        with torch.no_grad():
            for sample_idx, sample in enumerate(self.validation_dataloader):
                try:
                    if self.model.temporal_mode:
                        # Move sample dictionary to device
                        sample = self.to_device(sample)
                        
                        # FIX: Handle both LSTM and TCN temporal modes
                        if self.model.use_tcn:
                            # TCN returns (x0_pred, x1_pred, x2_pred)
                            x0_pred, x1_pred, x2_pred = self.model(sample)
                            
                            # Targets for all levels
                            target_x0 = sample['target_x0']
                            target_x1 = sample['target_x1']
                            target_x2 = sample['target_x2']
                            
                            # Compute errors for all levels
                            error_0 = self.crit(x0_pred, target_x0)
                            error_1 = self.crit(x1_pred, target_x1)
                            error_2 = self.crit(x2_pred, target_x2)
                        else:
                            # LSTM returns (x0_pred, x2_mean_pred)
                            x0_pred, x2_mean_pred = self.model(sample)
                            
                            # Target for 0-cells
                            target_x0 = sample['target_x0']
                            
                            # Compute error for 0-cell predictions
                            error_0 = self.crit(x0_pred, target_x0)
                            
                            # Initialize error_1 for consistency
                            error_1 = None
                            
                            # Process 2-cell errors for LSTM mode
                            if x2_mean_pred is not None:
                                b1 = sample['b1'].to_dense().squeeze(0) if sample['b1'].dim() > 2 else sample['b1'].to_dense()
                                b2 = sample['b2'].to_dense().squeeze(0) if sample['b2'].dim() > 2 else sample['b2'].to_dense()
                                
                                cell2_to_cell0 = {}
                                n_2cells = 5
                                
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
                            else:
                                error_2 = None
                            
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

                        # Process 1-cell errors (only for TCN)
                        if error_1 is not None:
                            if self.use_geometric_mean:
                                edge_errors = self.calculate_geometric_mean_error(error_1)
                                edge_sample_errors = torch.mean(edge_errors, dim=1)
                                all_1cell_errors.append(edge_sample_errors.cpu().numpy())
                            else:
                                edge_errors = error_1.mean(dim=2)
                                edge_sample_errors = edge_errors.max(dim=1)[0]
                                all_1cell_errors.append(edge_sample_errors.cpu().numpy())
                        
                        # Process 2-cell errors
                        if error_2 is not None:
                            if self.use_geometric_mean:
                                group_errors = self.calculate_geometric_mean_error(error_2)
                            else:
                                group_errors = error_2.mean(dim=2)
                            
                            group_errors_list.append(group_errors.cpu().numpy())
                            all_2cell_errors.append(torch.mean(group_errors, dim=1).cpu().numpy())
                    
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
                        all_1cell_errors.append(edge_sample_errors.cpu().numpy())
                    else:
                        edge_errors = error_1.mean(dim=2)
                        edge_sample_errors = edge_errors.max(dim=1)[0]
                        all_1cell_errors.append(edge_sample_errors.cpu().numpy())
                        
                    # Process 2-cell errors
                    if self.use_geometric_mean:
                        cell2_errors = self.calculate_geometric_mean_error(error_2)
                        cell2_sample_errors = torch.mean(cell2_errors, dim=1)
                        all_2cell_errors.append(cell2_sample_errors.cpu().numpy())
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
        
        # Handle case where 1-cell errors might be empty (LSTM mode)
        if len(all_1cell_errors) > 0:  # FIX: Check length instead of truthiness
            all_1cell_errors = np.concatenate(all_1cell_errors, axis=0)
        else:
            all_1cell_errors = None
            
        if len(all_2cell_errors) > 0:  # FIX: Check length instead of truthiness
            all_2cell_errors = np.concatenate(all_2cell_errors, axis=0)
        else:
            all_2cell_errors = None
        
        # Calculate global threshold based on selected method
        if self.threshold_method == "percentile":
            self.anomaly_threshold = np.percentile(all_0cell_errors, self.threshold_percentile)
            print(f"Calibrated global 0-cell anomaly threshold ({self.threshold_percentile}th percentile): {self.anomaly_threshold:.6f}")
            
            if all_1cell_errors is not None:
                self.cell1_threshold = np.percentile(all_1cell_errors, self.threshold_percentile)
                print(f"Calibrated global 1-cell anomaly threshold ({self.threshold_percentile}th percentile): {self.cell1_threshold:.6f}")
            else:
                self.cell1_threshold = 0.0
                print("No 1-cell errors collected, setting threshold to 0.0")
            
            if all_2cell_errors is not None:
                self.cell2_threshold = np.percentile(all_2cell_errors, self.threshold_percentile)
                print(f"Calibrated global 2-cell anomaly threshold ({self.threshold_percentile}th percentile): {self.cell2_threshold:.6f}")
            else:
                self.cell2_threshold = 0.0
                print("No 2-cell errors collected, setting threshold to 0.0")
        else:  # mean_sd method
            error_mean = np.mean(all_0cell_errors)
            error_std = np.std(all_0cell_errors)
            self.anomaly_threshold = error_mean + self.sd_multiplier * error_std
            print(f"Calibrated global 0-cell anomaly threshold (mean+{self.sd_multiplier}×SD): {self.anomaly_threshold:.6f}")
            print(f"Error distribution - Mean: {error_mean:.6f}, SD: {error_std:.6f}")
            
            if all_1cell_errors is not None:
                error1_mean = np.mean(all_1cell_errors)
                error1_std = np.std(all_1cell_errors)
                self.cell1_threshold = error1_mean + self.sd_multiplier * error1_std
                print(f"Calibrated global 1-cell anomaly threshold (mean+{self.sd_multiplier}×SD): {self.cell1_threshold:.6f}")
            else:
                self.cell1_threshold = 0.0
                print("No 1-cell errors collected, setting threshold to 0.0")
                
            if all_2cell_errors is not None:
                error2_mean = np.mean(all_2cell_errors)
                error2_std = np.std(all_2cell_errors)
                self.cell2_threshold = error2_mean + self.sd_multiplier * error2_std
                print(f"Calibrated global 2-cell anomaly threshold (mean+{self.sd_multiplier}×SD): {self.cell2_threshold:.6f}")
            else:
                self.cell2_threshold = 0.0
                print("No 2-cell errors collected, setting threshold to 0.0")
        
        # Calculate component-specific thresholds if enabled
        if self.use_component_thresholds and len(component_errors_list) > 0:  # FIX: Check length
            component_errors_array = np.concatenate(component_errors_list, axis=0)
            self.component_thresholds = []
            
            n_components = component_errors_array.shape[1]
            print(f"Calculating component-specific thresholds for {n_components} components...")
            
            for i in range(n_components):
                component_error = component_errors_array[:, i]
                if self.threshold_method == "percentile":
                    threshold = np.percentile(component_error, self.threshold_percentile)
                else:
                    mean = np.mean(component_error)
                    std = np.std(component_error)
                    threshold = mean + self.sd_multiplier * std
                self.component_thresholds.append(threshold)
            
            print(f"Component thresholds - min: {min(self.component_thresholds):.6f}, "
                  f"max: {max(self.component_thresholds):.6f}, "
                  f"mean: {np.mean(self.component_thresholds):.6f}")
        
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
                    
                    # FIX: Handle both LSTM and TCN temporal modes (same as calibrate_threshold)
                    if self.model.use_tcn:
                        # TCN returns (x0_pred, x1_pred, x2_pred)
                        x0_pred, x1_pred, x2_pred = self.model(sample)
                        target_x0 = sample['target_x0']
                        target_x1 = sample['target_x1']
                        target_x2 = sample['target_x2']
                        label = sample['label'].cpu().item()  # FIX: Convert CUDA tensor to Python scalar
                        
                        # Compute errors for all levels
                        error_0_raw = self.crit(x0_pred, target_x0)
                        error_1_raw = self.crit(x1_pred, target_x1)
                        error_2_raw = self.crit(x2_pred, target_x2)
                        
                        # Process errors with original logic
                        if self.use_geometric_mean:
                            error_0 = self.calculate_geometric_mean_error(error_0_raw)
                            error_1 = self.calculate_geometric_mean_error(error_1_raw)
                            error_2 = self.calculate_geometric_mean_error(error_2_raw)
                        else:
                            error_0 = error_0_raw.mean(dim=2)
                            error_1 = error_1_raw.mean(dim=2)
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
                        all_labels.append(label)  # Now label is a Python scalar, not CUDA tensor
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
                    else:
                        # LSTM mode (original temporal code)
                        x0_pred, x2_mean_pred = self.model(sample)
                        target_x0 = sample['target_x0']
                        label = sample['label'].cpu().item()  # FIX: Convert CUDA tensor to Python scalar
                        error_0_raw = self.crit(x0_pred, target_x0)
                        
                        # Basic temporal evaluation logic for LSTM
                        if self.use_geometric_mean:
                            error_0 = self.calculate_geometric_mean_error(error_0_raw)
                        else:
                            error_0 = error_0_raw.mean(dim=2)
                        
                        is_anomalous = torch.any(error_0[0] > self.anomaly_threshold).item()
                        all_preds.append(1 if is_anomalous else 0)
                        all_labels.append(label)  # Now label is a Python scalar, not CUDA tensor
                        component_errors.append(error_0.cpu().numpy())
                        edge_errors.append(np.zeros((1, 74)))  # Placeholder
                        cell2_errors.append(np.zeros((1, 5)))  # Placeholder
                        
                        preds_0cell.append(1 if is_anomalous else 0)
                        preds_1cell.append(0)
                        preds_2cell.append(0)
                        preds_hierarchical.append(1 if is_anomalous else 0)
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
            Minimum change in test F1 to qualify as improvement for early stopping
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
            
            # Log training loss to wandb
            wandb.log({"train_loss": train_loss, "epoch": epoch + 1})
            
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
                    
                    # Log test metrics to wandb
                    wandb.log({
                        "test_f1": test_f1,
                        "test_precision": test_precision,
                        "test_recall": test_recall,
                        "epoch": epoch + 1
                    })
                    
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