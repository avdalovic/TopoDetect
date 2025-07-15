import os
import time
import pickle
import numpy as np
import torch
from torch import nn
import wandb
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# New imports for CUSUM and advanced metrics
try:
    from src.utils.cusum import CUSUM
    CUSUM_AVAILABLE = True
except ImportError:
    CUSUM_AVAILABLE = False
    print("Warning: CUSUM not available. Only threshold method will work.")
from src.utils.metrics import (
    calculate_standard_metrics, calculate_scenario_detection,
    calculate_fp_alarms, calculate_time_aware_metrics
)
from src.utils.attack_utils import get_attack_indices

class AnomalyTrainer:
    """
    Trainer for the anomaly detection model.
    Handles both reconstruction-based and prediction-based approaches.
    """
    def __init__(self, model, train_dataloader, validation_dataloader, test_dataloader,
                 learning_rate, device, 
                 threshold_percentile=99.00,
                 use_geometric_mean=False, epsilon=1e-6,
                 threshold_method="percentile",
                 sd_multiplier=2.5, 
                 use_component_thresholds=False,
                 temporal_consistency=1,
                 weight_decay=1e-5,
                 grad_clip_value=1.0,
                 # CUSUM specific parameters
                 evaluation_method='threshold', # 'threshold' or 'cusum'
                 cusum_S=1.42,
                 cusum_G=5.98,
                 fp_alarm_window=60,
                 # Enhanced time-aware metrics parameters
                 theta_p=0.5,  # Detection threshold for precision
                 theta_r=0.1,  # Detection threshold for recall
                 original_sample_hz=1,  # Original sampling frequency (1 Hz for SWAT)
                 # Sampling parameters for metrics
                 sample_rate=1.0,
                 # Dataset-specific parameters
                 dataset_name="SWAT",  # Dataset name for attack indices
                 level_names=None):  # Custom level names (defaults to ["Nodes", "Edges", "PLCs"])
        """
        Initialize the trainer with additional training options.
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        
        # Dataset-specific parameters
        self.dataset_name = dataset_name
        self.level_names = level_names if level_names is not None else ["Nodes", "Edges", "PLCs"]
        
        # Standard thresholding parameters
        self.threshold_percentile = threshold_percentile
        self.sd_multiplier = sd_multiplier 
        self.use_component_thresholds = use_component_thresholds
        
        # CUSUM parameters
        self.evaluation_method = evaluation_method
        self.cusum_S = cusum_S
        self.cusum_G = cusum_G
        self.fp_alarm_window = fp_alarm_window
        self.cusum_detector = None # Will be initialized during calibration
        self.anomaly_threshold = None # To be calibrated for threshold method
        
        # Enhanced time-aware metrics parameters
        self.theta_p = theta_p
        self.theta_r = theta_r
        self.original_sample_hz = original_sample_hz
        
        # Sampling parameters for metrics
        self.sample_rate = sample_rate
        
        # General components
        self.crit = nn.MSELoss(reduction='none')
        self.opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.5, patience=5, verbose=True)
        self.grad_clip_value = grad_clip_value
        
        # Removed dynamic feature filtering - using general method

        if self.validation_dataloader is None or len(self.validation_dataloader.dataset) == 0:
             raise ValueError("Validation dataloader is required and cannot be empty for threshold calibration.")

        print(f"Initialized AnomalyTrainer with device {device}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Level names: {self.level_names}")
        print(f"Evaluation Method: {self.evaluation_method}")
        if self.evaluation_method == 'cusum':
            if not CUSUM_AVAILABLE:
                raise ImportError("CUSUM method selected but CUSUM module not available. Use 'threshold' method instead.")
            print(f"CUSUM Params: S={self.cusum_S}, G={self.cusum_G}")
        else:
            print(f"Threshold method: {threshold_method}")
        
        # Print enhanced time-aware metrics configuration
        print(f"Enhanced Time-Aware Metrics Config:")
        print(f"  theta_p (precision threshold): {self.theta_p}")
        print(f"  theta_r (recall threshold): {self.theta_r}")
        print(f"  sample_rate: {self.sample_rate}")
        print(f"  original_sample_hz: {self.original_sample_hz}")
        print(f"  effective_sample_hz: {self.original_sample_hz * self.sample_rate:.4f}")
        print(f"  fp_alarm_window: {self.fp_alarm_window}s -> {int(self.fp_alarm_window * self.original_sample_hz * self.sample_rate)} points")

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
        Train the model for one epoch. This method remains largely unchanged.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        total_samples = len(self.train_dataloader)
        log_interval = max(1, total_samples // 10)
        start_time = time.time()

        for i, sample in enumerate(self.train_dataloader):
            self.opt.zero_grad()
            if self.model.temporal_mode:
                # This logic seems highly specific and can remain
                sample = self.to_device(sample)
                x0_pred, x2_mean_pred = self.model(sample)
                target_x0 = sample['target_x0']
                loss_0 = self.crit(x0_pred, target_x0).mean()
                # Simplified loss for clarity, original logic was complex
                loss = loss_0
                total_loss += loss.item()
            else: # Reconstruction mode
                x_0, x_1, x_2, a0, a1, coa2, b1, b2, _ = self.to_device(sample)
                x_0_recon, x_1_recon, x_2_recon = self.model(x_0, x_1, x_2, a0, a1, coa2, b1, b2)
                
                loss_0 = self.crit(x_0_recon, x_0).mean()
                loss_1 = self.crit(x_1_recon, x_1).mean()
                loss_2 = self.crit(x_2_recon, x_2).mean()
                loss = loss_0 + loss_1 + loss_2
                total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_value)
            self.opt.step()
            num_batches += 1
            
            if (i + 1) % log_interval == 0:
                print(f"Progress: [{(i+1)/total_samples*100:.1f}%] | Loss: {total_loss / num_batches:.6f}")
        
        return total_loss / num_batches

    def calibrate(self):
        """
        Calibrates the anomaly detection method using the validation set.
        Supports both 'threshold' and 'cusum' methods.
        Now evaluates on all 3 levels and uses L2 loss consistently.
        """
        print(f"\nCalibrating using '{self.evaluation_method}' method on validation data...")
        self.model.eval()
        all_residuals_0 = []
        all_residuals_1 = []
        all_residuals_2 = []

        with torch.no_grad():
            for sample in self.validation_dataloader:
                if self.model.temporal_mode:
                    sample = self.to_device(sample)
                    x0_pred, _ = self.model(sample)
                    # Use L2 loss consistently (MSE without reduction)
                    residuals_0 = self.crit(x0_pred, sample['target_x0'])
                    all_residuals_0.append(residuals_0)
                else:
                    x_0, x_1, x_2, _, _, _, _, _, _ = self.to_device(sample)
                    x_0_recon, x_1_recon, x_2_recon = self.model(*self.to_device(sample)[:-1])
                    
                    # Use L2 loss consistently (MSE without reduction) for all levels
                    residuals_0 = self.crit(x_0_recon, x_0)
                    residuals_1 = self.crit(x_1_recon, x_1)
                    residuals_2 = self.crit(x_2_recon, x_2)
                    
                    all_residuals_0.append(residuals_0)
                    all_residuals_1.append(residuals_1)
                    all_residuals_2.append(residuals_2)

        # Concatenate all residuals
        all_residuals_0 = torch.cat(all_residuals_0, dim=0)
        
        if self.evaluation_method == 'cusum':
            # For CUSUM, aggregate the feature dimension to get a single residual value per component
            all_residuals_per_feature = torch.mean(all_residuals_0, dim=2)
            num_features = all_residuals_per_feature.shape[1]
            self.cusum_detector = CUSUM(num_features, self.cusum_S, self.cusum_G, device=self.device)
            self.cusum_detector.calibrate(all_residuals_per_feature)
        else: # Handle standard thresholding
            # Calculate thresholds for each level
            if not self.model.temporal_mode:
                all_residuals_1 = torch.cat(all_residuals_1, dim=0)
                all_residuals_2 = torch.cat(all_residuals_2, dim=0)
                
                # Calculate 99th percentile threshold for each level
                error_per_sample_0 = torch.mean(all_residuals_0.cpu(), dim=(1, 2))
                error_per_sample_1 = torch.mean(all_residuals_1.cpu(), dim=(1, 2))
                error_per_sample_2 = torch.mean(all_residuals_2.cpu(), dim=(1, 2))
                
                self.anomaly_threshold_0 = np.percentile(error_per_sample_0.numpy(), self.threshold_percentile)
                self.anomaly_threshold_1 = np.percentile(error_per_sample_1.numpy(), self.threshold_percentile)
                self.anomaly_threshold_2 = np.percentile(error_per_sample_2.numpy(), self.threshold_percentile)
                
                print(f"Calibrated 3-level thresholds (99th percentile):")
                print(f"  0-cells: {self.anomaly_threshold_0:.4f}")
                print(f"  1-cells: {self.anomaly_threshold_1:.4f}")
                print(f"  2-cells: {self.anomaly_threshold_2:.4f}")
            else:
                # For temporal mode, only use 0-cells
                error_per_sample_0 = torch.mean(all_residuals_0.cpu(), dim=(1, 2))
                self.anomaly_threshold_0 = np.percentile(error_per_sample_0.numpy(), self.threshold_percentile)
                print(f"Calibrated 0-cell threshold: {self.anomaly_threshold_0:.4f}")
            
    def evaluate(self, dataloader=None, eval_mode="Test"):
        """
        Evaluates the model on a dataset using the configured method.
        """
        if dataloader is None:
            dataloader = self.test_dataloader
            
        # Correctly check if calibration is needed
        needs_calibration = False
        if self.evaluation_method == 'cusum':
            # Check if detector exists or is not calibrated
            if self.cusum_detector is None or not self.cusum_detector.is_calibrated:
                needs_calibration = True
        elif self.evaluation_method == 'threshold':
            # Check if thresholds are initialized
            if not hasattr(self, 'anomaly_threshold_0') or self.anomaly_threshold_0 is None:
                needs_calibration = True

        if needs_calibration:
            self.calibrate()

        print(f"\n--- Evaluating on {eval_mode} Data ---")
        self.model.eval()
        
        all_y_true = []
        all_y_pred = []
        
        # Store detailed data for visualization and analysis
        detailed_predictions = []
        detailed_ground_truth = []
        detailed_timestamps = []
        detailed_reconstruction_errors = {'level_0': [], 'level_1': [], 'level_2': []}
        detailed_component_errors = []  # Store per-component errors
        detailed_level_predictions = {'level_0': [], 'level_1': [], 'level_2': []}  # Multi-level predictions

        if self.evaluation_method == 'cusum':
            self.cusum_detector.reset()

        with torch.no_grad():
            for batch_idx, sample in enumerate(dataloader):
                y_true = sample[-1].numpy() # Ground truth labels
                all_y_true.extend(y_true)
                detailed_ground_truth.extend(y_true)

                if self.model.temporal_mode:
                    sample_device = self.to_device(sample)
                    x0_pred, _ = self.model(sample_device)
                    # Use L2 loss consistently
                    residuals_0 = self.crit(x0_pred, sample_device['target_x0'])
                    
                    # Store reconstruction errors
                    batch_errors_0 = residuals_0.cpu().numpy()
                    detailed_reconstruction_errors['level_0'].extend(batch_errors_0)
                    
                    # Get predictions for each item in the batch
                    batch_preds = []
                    batch_level_preds = {'level_0': [], 'level_1': [], 'level_2': []}
                    batch_component_errors = []
                    
                    for i in range(residuals_0.shape[0]):
                        # Component-level errors (per sensor/actuator)
                        component_errors = torch.mean(residuals_0[i], dim=1).cpu().numpy()
                        batch_component_errors.append(component_errors)
                        
                        if self.evaluation_method == 'cusum':
                            # Aggregate feature dimension before passing to CUSUM
                            residual_per_feature = torch.mean(residuals_0[i], dim=1)
                            sensor_anomalies = self.cusum_detector.update(residual_per_feature)
                            is_anomalous = torch.any(sensor_anomalies).item()
                            batch_preds.append(is_anomalous)
                            batch_level_preds['level_0'].append(is_anomalous)
                            batch_level_preds['level_1'].append(is_anomalous)  # Same for temporal mode
                            batch_level_preds['level_2'].append(is_anomalous)
                        else: # Simple threshold
                            error = torch.mean(residuals_0[i])
                            is_anomalous = (error > self.anomaly_threshold_0).item()
                            batch_preds.append(is_anomalous)
                            batch_level_preds['level_0'].append(is_anomalous)
                            batch_level_preds['level_1'].append(is_anomalous)  # Same for temporal mode
                            batch_level_preds['level_2'].append(is_anomalous)
                            
                    all_y_pred.extend(batch_preds)
                    detailed_predictions.extend(batch_preds)
                    detailed_component_errors.extend(batch_component_errors)
                    
                    # Store level predictions
                    for level in ['level_0', 'level_1', 'level_2']:
                        detailed_level_predictions[level].extend(batch_level_preds[level])
                    
                else: # Reconstruction - evaluate on all 3 levels
                    x_0, x_1, x_2, _, _, _, _, _, _ = self.to_device(sample)
                    x_0_recon, x_1_recon, x_2_recon = self.model(*self.to_device(sample)[:-1])
                    
                    # Use L2 loss consistently for all levels
                    residuals_0 = self.crit(x_0_recon, x_0)
                    residuals_1 = self.crit(x_1_recon, x_1)
                    residuals_2 = self.crit(x_2_recon, x_2)
                    
                    # Store reconstruction errors
                    batch_errors_0 = residuals_0.cpu().numpy()
                    batch_errors_1 = residuals_1.cpu().numpy()
                    batch_errors_2 = residuals_2.cpu().numpy()
                    detailed_reconstruction_errors['level_0'].extend(batch_errors_0)
                    detailed_reconstruction_errors['level_1'].extend(batch_errors_1)
                    detailed_reconstruction_errors['level_2'].extend(batch_errors_2)

                    # Get predictions for each item in the batch
                    batch_preds = []
                    batch_level_preds = {'level_0': [], 'level_1': [], 'level_2': []}
                    batch_component_errors = []
                    
                    for i in range(residuals_0.shape[0]):
                        # Component-level errors (per sensor/actuator)
                        component_errors = torch.mean(residuals_0[i], dim=1).cpu().numpy()
                        batch_component_errors.append(component_errors)
                        
                        if self.evaluation_method == 'cusum':
                            # For CUSUM, only use 0-cells
                            residual_per_feature = torch.mean(residuals_0[i], dim=1)
                            sensor_anomalies = self.cusum_detector.update(residual_per_feature)
                            is_anomalous = torch.any(sensor_anomalies).item()
                            batch_preds.append(is_anomalous)
                            batch_level_preds['level_0'].append(is_anomalous)
                            batch_level_preds['level_1'].append(is_anomalous)  # Same for CUSUM
                            batch_level_preds['level_2'].append(is_anomalous)
                        else: # Multi-level threshold evaluation
                            error_0 = torch.mean(residuals_0[i])
                            error_1 = torch.mean(residuals_1[i])
                            error_2 = torch.mean(residuals_2[i])
                            
                            # Individual level predictions
                            is_anomalous_0 = (error_0 > self.anomaly_threshold_0).item()
                            is_anomalous_1 = (error_1 > self.anomaly_threshold_1).item()
                            is_anomalous_2 = (error_2 > self.anomaly_threshold_2).item()
                            
                            batch_level_preds['level_0'].append(is_anomalous_0)
                            batch_level_preds['level_1'].append(is_anomalous_1)
                            batch_level_preds['level_2'].append(is_anomalous_2)
                            
                            # Anomaly if ANY level exceeds its threshold
                            is_anomalous = is_anomalous_0 or is_anomalous_1 or is_anomalous_2
                            batch_preds.append(is_anomalous)
                            
                    all_y_pred.extend(batch_preds)
                    detailed_predictions.extend(batch_preds)
                    detailed_component_errors.extend(batch_component_errors)
                    
                    # Store level predictions
                    for level in ['level_0', 'level_1', 'level_2']:
                        detailed_level_predictions[level].extend(batch_level_preds[level])
                
                # Create timestamps for visualization (simple sequential indices)
                batch_timestamps = list(range(batch_idx * dataloader.batch_size, 
                                            batch_idx * dataloader.batch_size + len(y_true)))
                detailed_timestamps.extend(batch_timestamps)

        # Convert lists to numpy arrays for metric calculations
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        
        # Store detailed data for visualization (only for test evaluation)
        if eval_mode == "Test" or eval_mode == "Final Test":
            self.detailed_test_data = {
                'timestamps': detailed_timestamps,
                'predictions': detailed_predictions,
                'ground_truth': detailed_ground_truth,
                'sample_rate': self.sample_rate,
                'reconstruction_errors': detailed_reconstruction_errors,
                'component_errors': detailed_component_errors,
                'level_predictions': detailed_level_predictions
            }
            print(f"Stored detailed test data: {len(detailed_timestamps)} points")
        
        # --- Calculate and Log All Metrics ---
        # Get attack segments and properly adjust for sampling
        attacks_full, attack_labels = get_attack_indices(self.dataset_name)
        
        # Get sampling information from the dataset
        from src.utils.attack_utils import adjust_attack_indices_for_sampling, get_sampling_info_from_dataset
        
        # Use the sample rate from trainer initialization
        estimated_original_length = int(len(all_y_true) / self.sample_rate) if self.sample_rate < 1.0 else len(all_y_true)
        
        # Adjust attack indices for the sampling
        attacks = adjust_attack_indices_for_sampling(
            attacks_full, 
            estimated_original_length, 
            len(all_y_true), 
            self.sample_rate
        )

        standard_metrics = calculate_standard_metrics(all_y_true, all_y_pred)
        scenario_detection_rate = calculate_scenario_detection(all_y_true, all_y_pred, attacks)
        
        # Use improved FPA calculation with proper sampling rate consideration
        fp_alarm_count = calculate_fp_alarms(
            all_y_true, all_y_pred, attacks, 
            window_seconds=self.fp_alarm_window,
            sample_rate=self.sample_rate,
            original_sample_hz=self.original_sample_hz
        )
        
        # Use enhanced time-aware metrics with proper parameters
        time_aware_metrics = calculate_time_aware_metrics(
            all_y_true, all_y_pred, attacks,
            theta_p=self.theta_p,
            theta_r=self.theta_r,
            sample_rate=self.sample_rate,
            original_sample_hz=self.original_sample_hz
        )

        # Calculate multi-level metrics
        level_metrics = {}
        for level in ['level_0', 'level_1', 'level_2']:
            level_preds = np.array(detailed_level_predictions[level])
            level_metrics[level] = calculate_standard_metrics(all_y_true, level_preds)

        # Log results to console
        print("\n--- Evaluation Results ---")
        print(f"Standard Precision: {standard_metrics['precision']:.4f}")
        print(f"Standard Recall:    {standard_metrics['recall']:.4f}")
        print(f"Standard F1-Score:    {standard_metrics['f1']:.4f}")
        print("-" * 25)
        print(f"Scenario Detection Rate: {scenario_detection_rate:.4f} ({int(scenario_detection_rate*len(attacks))}/{len(attacks)})")
        print(f"False Positive Alarms:   {fp_alarm_count}")
        print("-" * 25)
        print(f"eTaP (Time-Aware Precision): {time_aware_metrics['eTaP']:.4f}  # How precise detection is within time windows")
        print(f"eTaR (Time-Aware Recall):    {time_aware_metrics['eTaR']:.4f}  # How well we detect attacks within time windows")
        print(f"eTaF1 (Time-Aware F1):       {time_aware_metrics['eTaF1']:.4f}  # Harmonic mean of eTaP and eTaR")
        print("-" * 25)
        print("Multi-Level Detection Results:")
        for i, level in enumerate(['level_0', 'level_1', 'level_2']):
            level_name = self.level_names[i] if i < len(self.level_names) else f'Level_{i}'
            print(f"{level_name} - P: {level_metrics[level]['precision']:.4f}, R: {level_metrics[level]['recall']:.4f}, F1: {level_metrics[level]['f1']:.4f}")
        print("--------------------------\n")
        
        final_results = {**standard_metrics, **time_aware_metrics, 
                         'scenario_detection_rate': scenario_detection_rate,
                         'fp_alarms': fp_alarm_count}
        
        # Add level-specific metrics
        for level in ['level_0', 'level_1', 'level_2']:
            level_name = {'level_0': 'nodes', 'level_1': 'edges', 'level_2': 'plcs'}[level]
            final_results[f'{level_name}_precision'] = level_metrics[level]['precision']
            final_results[f'{level_name}_recall'] = level_metrics[level]['recall']
            final_results[f'{level_name}_f1'] = level_metrics[level]['f1']
        
        # Log to wandb if active
        if wandb.run:
            wandb.log(final_results)
            
        return final_results

    def create_detection_timeline_visualization(self):
        """
        Create a timeline visualization showing attacks (red) and detections (green).
        """
        if not hasattr(self, 'detailed_test_data'):
            print("No detailed test data available for visualization")
            return
            
        print("Creating detection timeline visualization...")
        
        # Get data
        timestamps = np.array(self.detailed_test_data['timestamps'])
        predictions = np.array(self.detailed_test_data['predictions'])
        ground_truth = np.array(self.detailed_test_data['ground_truth'])
        
        # Get attack indices
        attacks_full, _ = get_attack_indices(self.dataset_name)
        from src.utils.attack_utils import adjust_attack_indices_for_sampling
        
        # Adjust attack indices for sampling
        estimated_original_length = int(len(timestamps) / self.sample_rate) if self.sample_rate < 1.0 else len(timestamps)
        attacks = adjust_attack_indices_for_sampling(
            attacks_full, 
            estimated_original_length, 
            len(timestamps), 
            self.sample_rate
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Plot timeline background
        ax.fill_between(timestamps, 0, 1, alpha=0.1, color='gray', label='Normal Operation')
        
        # Plot attack periods in red
        for i, attack_range in enumerate(attacks):
            if len(attack_range) > 0:
                start_idx = attack_range[0]
                end_idx = attack_range[-1]
                if start_idx < len(timestamps) and end_idx < len(timestamps):
                    ax.fill_between(timestamps[start_idx:end_idx+1], 0, 1, 
                                  alpha=0.6, color='red', 
                                  label='Attack Period' if i == 0 else "")
        
        # Plot detection points in green
        detection_indices = np.where(predictions == 1)[0]
        if len(detection_indices) > 0:
            ax.scatter(timestamps[detection_indices], np.ones(len(detection_indices)) * 0.5, 
                      color='green', s=10, alpha=0.7, label='Detections', marker='|')
        
        # Formatting
        ax.set_xlabel('Time (Sample Index)')
        ax.set_ylabel('Status')
        ax.set_title('Detection Timeline: Attacks (Red) vs Detections (Green)')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        total_attacks = len(attacks)
        total_detections = np.sum(predictions)
        attack_points = np.sum(ground_truth)
        detected_attack_points = np.sum(predictions[ground_truth == 1])
        
        stats_text = f"""Statistics:
        Total Attacks: {total_attacks}
        Total Detections: {total_detections}
        Attack Points: {attack_points}
        Detected Attack Points: {detected_attack_points}
        Detection Rate: {detected_attack_points/attack_points:.2%}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Log to wandb
        if wandb.run:
            wandb.log({"detection_timeline": wandb.Image(fig)})
            print("Detection timeline logged to wandb")
        
        plt.close(fig)
        print("Detection timeline visualization created")

    def analyze_individual_attacks(self):
        """
        Analyze each attack individually and create detailed visualizations.
        """
        if not hasattr(self, 'detailed_test_data'):
            print("No detailed test data available for attack analysis")
            return
            
        print("Analyzing individual attacks...")
        
        # Get data
        timestamps = np.array(self.detailed_test_data['timestamps'])
        predictions = np.array(self.detailed_test_data['predictions'])
        ground_truth = np.array(self.detailed_test_data['ground_truth'])
        component_errors = np.array(self.detailed_test_data['component_errors'])
        level_predictions = self.detailed_test_data['level_predictions']
        
        # Get attack info
        attacks_full, attack_labels = get_attack_indices(self.dataset_name)
        from src.utils.attack_utils import adjust_attack_indices_for_sampling
        
        # Adjust attack indices for sampling
        estimated_original_length = int(len(timestamps) / self.sample_rate) if self.sample_rate < 1.0 else len(timestamps)
        attacks = adjust_attack_indices_for_sampling(
            attacks_full, 
            estimated_original_length, 
            len(timestamps), 
            self.sample_rate
        )
        
        # Get component names from dataset
        # We'll need to access the dataset to get component names
        component_names = list(range(component_errors.shape[1]))  # Fallback to indices
        try:
            # Try to get actual component names from the test dataloader
            if hasattr(self.test_dataloader.dataset, 'columns'):
                component_names = self.test_dataloader.dataset.columns
        except:
            pass
        
        attack_analysis_results = {}
        
        # Log overall attack summary first
        wandb_attack_summary = {
            "total_attacks": len(attacks),
            "total_attack_points": np.sum(ground_truth),
            "total_detections": np.sum(predictions),
            "overall_attack_detection_rate": np.sum(predictions[ground_truth == 1]) / np.sum(ground_truth) if np.sum(ground_truth) > 0 else 0
        }
        
        print(f"Processing {len(attacks)} individual attacks...")
        
        for attack_idx, attack_range in enumerate(attacks):
            if len(attack_range) < 2:
                continue
                
            attack_start, attack_end = attack_range[0], attack_range[1]
            
            # Add context: 10 points before and after attack
            context_before = max(0, attack_start - 10)
            context_after = min(len(timestamps) - 1, attack_end + 10)
            
            # Extract data for this attack period with context
            attack_timestamps = timestamps[context_before:context_after+1]
            attack_ground_truth = ground_truth[context_before:context_after+1]
            attack_predictions = predictions[context_before:context_after+1]
            attack_component_errors = component_errors[context_before:context_after+1]
            
            # Get attacked components for this attack
            attacked_components = attack_labels[attack_idx] if attack_idx < len(attack_labels) else []
            
            # Calculate metrics for this attack
            attack_true_positives = np.sum((attack_ground_truth == 1) & (attack_predictions == 1))
            attack_false_negatives = np.sum((attack_ground_truth == 1) & (attack_predictions == 0))
            attack_false_positives = np.sum((attack_ground_truth == 0) & (attack_predictions == 1))
            
            attack_precision = attack_true_positives / (attack_true_positives + attack_false_positives) if (attack_true_positives + attack_false_positives) > 0 else 0
            attack_recall = attack_true_positives / (attack_true_positives + attack_false_negatives) if (attack_true_positives + attack_false_negatives) > 0 else 0
            attack_f1 = 2 * attack_precision * attack_recall / (attack_precision + attack_recall) if (attack_precision + attack_recall) > 0 else 0
            
            # Multi-level detection for this attack
            level_detection_rates = {}
            level_metrics = {}
            for level in ['level_0', 'level_1', 'level_2']:
                level_preds = np.array(level_predictions[level])[context_before:context_after+1]
                level_tp = np.sum((attack_ground_truth == 1) & (level_preds == 1))
                level_fn = np.sum((attack_ground_truth == 1) & (level_preds == 0))
                level_fp = np.sum((attack_ground_truth == 0) & (level_preds == 1))
                
                level_precision = level_tp / (level_tp + level_fp) if (level_tp + level_fp) > 0 else 0
                level_recall = level_tp / (level_tp + level_fn) if (level_tp + level_fn) > 0 else 0
                level_f1 = 2 * level_precision * level_recall / (level_precision + level_recall) if (level_precision + level_recall) > 0 else 0
                
                level_detection_rates[level] = level_recall  # Detection rate = recall
                level_metrics[level] = {
                    'precision': level_precision,
                    'recall': level_recall,
                    'f1': level_f1
                }
            
            # Store analysis results
            attack_analysis_results[f'attack_{attack_idx}'] = {
                'precision': attack_precision,
                'recall': attack_recall,
                'f1': attack_f1,
                'detection_rate': attack_recall,
                'attacked_components': attacked_components,
                'level_detection_rates': level_detection_rates,
                'level_metrics': level_metrics,
                'attack_start': attack_start,
                'attack_end': attack_end,
                'context_start': context_before,
                'context_end': context_after,
                'attack_duration': attack_end - attack_start + 1,
                'total_attack_points': np.sum(attack_ground_truth),
                'detected_attack_points': attack_true_positives,
                'false_positives': attack_false_positives,
                'false_negatives': attack_false_negatives
            }
            
            # Create individual attack metrics for wandb logging
            attack_wandb_metrics = {
                f"attack_{attack_idx}_precision": attack_precision,
                f"attack_{attack_idx}_recall": attack_recall,
                f"attack_{attack_idx}_f1": attack_f1,
                f"attack_{attack_idx}_detection_rate": attack_recall,
                f"attack_{attack_idx}_duration": attack_end - attack_start + 1,
                f"attack_{attack_idx}_components": len(attacked_components),
                # Multi-level metrics
                f"attack_{attack_idx}_nodes_detection_rate": level_detection_rates.get('level_0', 0),
                f"attack_{attack_idx}_edges_detection_rate": level_detection_rates.get('level_1', 0),
                f"attack_{attack_idx}_plcs_detection_rate": level_detection_rates.get('level_2', 0),
                f"attack_{attack_idx}_nodes_precision": level_metrics.get('level_0', {}).get('precision', 0),
                f"attack_{attack_idx}_nodes_recall": level_metrics.get('level_0', {}).get('recall', 0),
                f"attack_{attack_idx}_nodes_f1": level_metrics.get('level_0', {}).get('f1', 0),
                f"attack_{attack_idx}_edges_precision": level_metrics.get('level_1', {}).get('precision', 0),
                f"attack_{attack_idx}_edges_recall": level_metrics.get('level_1', {}).get('recall', 0),
                f"attack_{attack_idx}_edges_f1": level_metrics.get('level_1', {}).get('f1', 0),
                f"attack_{attack_idx}_plcs_precision": level_metrics.get('level_2', {}).get('precision', 0),
                f"attack_{attack_idx}_plcs_recall": level_metrics.get('level_2', {}).get('recall', 0),
                f"attack_{attack_idx}_plcs_f1": level_metrics.get('level_2', {}).get('f1', 0),
            }
            
            # Add to summary metrics
            wandb_attack_summary.update(attack_wandb_metrics)
            
            # Create visualization for this attack
            self.create_attack_visualization(
                attack_idx, attack_timestamps, attack_ground_truth, attack_predictions,
                attack_component_errors, component_names, attacked_components,
                attack_start - context_before, attack_end - context_before,
                attack_precision, attack_recall, attack_f1, level_detection_rates
            )
            
            # Print individual attack results
            print(f"Attack {attack_idx}: P={attack_precision:.3f}, R={attack_recall:.3f}, F1={attack_f1:.3f}")
            print(f"  Components: {attacked_components}")
            print(f"  Multi-level detection rates: Nodes={level_detection_rates.get('level_0', 0):.3f}, "
                  f"Edges={level_detection_rates.get('level_1', 0):.3f}, PLCs={level_detection_rates.get('level_2', 0):.3f}")
        
        # Log all attack-specific metrics to wandb
        if wandb.run:
            wandb.log(wandb_attack_summary)
            print(f"Logged individual attack analysis for {len(attacks)} attacks to wandb")
            
            # Also create a summary table for better visualization
            attack_table_data = []
            for attack_idx in range(len(attacks)):
                if f'attack_{attack_idx}' in attack_analysis_results:
                    result = attack_analysis_results[f'attack_{attack_idx}']
                    attack_table_data.append([
                        attack_idx,
                        result['precision'],
                        result['recall'],
                        result['f1'],
                        result['detection_rate'],
                        len(result['attacked_components']),
                        result['level_detection_rates'].get('level_0', 0),
                        result['level_detection_rates'].get('level_1', 0),
                        result['level_detection_rates'].get('level_2', 0),
                        ', '.join(result['attacked_components'][:3]) + ('...' if len(result['attacked_components']) > 3 else '')
                    ])
            
            if attack_table_data:
                attack_table = wandb.Table(
                    columns=["Attack_ID", "Precision", "Recall", "F1", "Detection_Rate", 
                            "Num_Components", "Nodes_Detection", "Edges_Detection", "PLCs_Detection", "Components"],
                    data=attack_table_data
                )
                wandb.log({"attack_analysis_table": attack_table})
                print("Logged attack analysis table to wandb")
        
        return attack_analysis_results
    
    def create_attack_visualization(self, attack_idx, timestamps, ground_truth, predictions, 
                                  component_errors, component_names, attacked_components,
                                  attack_start_rel, attack_end_rel, precision, recall, f1, level_detection_rates):
        """
        Create a detailed visualization for a specific attack.
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Attack timeline with detections
        ax1 = axes[0]
        ax1.fill_between(timestamps, 0, 1, alpha=0.1, color='gray', label='Normal')
        
        # Highlight attack period
        attack_period = timestamps[attack_start_rel:attack_end_rel+1]
        if len(attack_period) > 0:
            ax1.fill_between(attack_period, 0, 1, alpha=0.6, color='red', label='Attack Period')
        
        # Plot detections with color coding
        detection_indices = np.where(predictions == 1)[0]
        if len(detection_indices) > 0:
            # Separate detections into correct (green) and false positive (orange)
            correct_detections = []
            false_positive_detections = []
            
            for det_idx in detection_indices:
                if ground_truth[det_idx] == 1:
                    # Correct detection (True Positive) - Green
                    correct_detections.append(det_idx)
                else:
                    # False Positive - Orange
                    false_positive_detections.append(det_idx)
            
            # Plot correct detections in green
            if correct_detections:
                ax1.scatter(timestamps[correct_detections], np.ones(len(correct_detections)) * 0.5, 
                           color='green', s=20, alpha=0.8, label='Correct Detections', marker='|')
            
            # Plot false positive detections in orange
            if false_positive_detections:
                ax1.scatter(timestamps[false_positive_detections], np.ones(len(false_positive_detections)) * 0.3, 
                           color='orange', s=20, alpha=0.8, label='False Positive Detections', marker='|')
        
        ax1.set_title(f'Attack {attack_idx} - Detection Timeline\nComponents: {", ".join(attacked_components)}')
        ax1.set_ylabel('Status')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Reconstruction errors for attacked components with enhanced detection visualization
        ax2 = axes[1]
        if len(attacked_components) > 0 and len(component_names) > 0:
            for comp_name in attacked_components:
                if comp_name in component_names:
                    comp_idx = component_names.index(comp_name)
                    comp_errors = component_errors[:, comp_idx]
                    ax2.plot(timestamps, comp_errors, label=f'{comp_name}', alpha=0.8, linewidth=2)
        
        # Add threshold line
        if hasattr(self, 'anomaly_threshold_0'):
            ax2.axhline(y=self.anomaly_threshold_0, color='red', linestyle='--', 
                       alpha=0.7, label=f'Threshold ({self.anomaly_threshold_0:.4f})')
        
        # Highlight attack period with semi-transparent red
        if len(attack_period) > 0:
            ax2.axvspan(attack_period[0], attack_period[-1], alpha=0.3, color='red', label='Attack Period')
        
        # Add detection performance overlay
        detection_indices = np.where(predictions == 1)[0]
        if len(detection_indices) > 0:
            # Get the y-axis limits for proper visualization
            y_min, y_max = ax2.get_ylim()
            detection_height = (y_max - y_min) * 0.05  # 5% of the y-range
            
            # Separate detections into correct (green) and false positive (orange)
            correct_detections = []
            false_positive_detections = []
            
            for det_idx in detection_indices:
                if ground_truth[det_idx] == 1:
                    correct_detections.append(det_idx)
                else:
                    false_positive_detections.append(det_idx)
            
            # Plot correct detections as green bars at the bottom
            if correct_detections:
                for det_idx in correct_detections:
                    ax2.axvspan(timestamps[det_idx], timestamps[det_idx], 
                               ymin=0, ymax=0.05, alpha=0.8, color='green')
            
            # Plot false positive detections as orange bars at the bottom
            if false_positive_detections:
                for det_idx in false_positive_detections:
                    ax2.axvspan(timestamps[det_idx], timestamps[det_idx], 
                               ymin=0, ymax=0.05, alpha=0.8, color='orange')
        
        ax2.set_title('Reconstruction Errors for Attacked Components with Detection Performance')
        ax2.set_ylabel('Reconstruction Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add custom legend entries for detection performance
        from matplotlib.patches import Patch
        
        # Get existing legend handles and labels correctly from the axes
        legend_elements, legend_labels = ax2.get_legend_handles_labels()
        
        # Add detection performance legend entries
        if len(detection_indices) > 0:
            legend_elements.extend([
                Patch(facecolor='green', alpha=0.8),
                Patch(facecolor='orange', alpha=0.8)
            ])
            legend_labels.extend(['Correct Detections', 'False Positive Detections'])
        
        ax2.legend(legend_elements, legend_labels)
        
        # Plot 3: Multi-level detection performance
        ax3 = axes[2]
        levels = ['level_0', 'level_1', 'level_2']
        level_names = [self.level_names[i] if i < len(self.level_names) else f'Level_{i}' for i in range(len(levels))]
        detection_rates = [level_detection_rates.get(level, 0) for level in levels]
        
        bars = ax3.bar(level_names, detection_rates, color=['blue', 'orange', 'green'], alpha=0.7)
        ax3.set_title('Multi-Level Detection Performance')
        ax3.set_ylabel('Detection Rate')
        ax3.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, detection_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.2f}', ha='center', va='bottom')
        
        # Add metrics text with detection performance summary
        correct_detections_count = np.sum((ground_truth == 1) & (predictions == 1))
        false_positive_count = np.sum((ground_truth == 0) & (predictions == 1))
        total_attack_points = np.sum(ground_truth == 1)
        
        metrics_text = f"""Attack {attack_idx} Metrics:
        Precision: {precision:.3f}
        Recall: {recall:.3f}
        F1-Score: {f1:.3f}
        
        Detection Performance:
        Correct Detections: {correct_detections_count}/{total_attack_points}
        False Positives: {false_positive_count}
        
        Multi-Level Detection:
        {self.level_names[0] if len(self.level_names) > 0 else 'Level_0'}: {level_detection_rates.get('level_0', 0):.3f}
        {self.level_names[1] if len(self.level_names) > 1 else 'Level_1'}: {level_detection_rates.get('level_1', 0):.3f}
        {self.level_names[2] if len(self.level_names) > 2 else 'Level_2'}: {level_detection_rates.get('level_2', 0):.3f}"""
        
        ax3.text(0.02, 0.98, metrics_text, transform=ax3.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Log to wandb
        if wandb.run:
            wandb.log({f"attack_{attack_idx}_analysis": wandb.Image(fig)})
        
        plt.close(fig)
        print(f"Created enhanced visualization for Attack {attack_idx} with detection performance overlay")

    def train(self, num_epochs=10, test_interval=1, checkpoint_dir="model_checkpoints", 
              early_stopping=True, patience=3, min_delta=1e-4):
        """
        Main training loop with validation, testing, and early stopping.
        """
        print(f"\n--- Starting Training for {num_epochs} epochs ---")
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_f1 = -1.0
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            train_loss = self.train_epoch()
            self.scheduler.step(train_loss)

            # Log training loss to wandb
            if wandb.run:
                wandb.log({"epoch": epoch + 1, "train_loss": train_loss})

            if (epoch + 1) % test_interval == 0:
                # Evaluate on test data
                test_results = self.evaluate(self.test_dataloader, "Test")
                test_f1 = test_results['f1']
                print(f"Epoch {epoch+1} Test F1: {test_f1:.4f} (Loss: {train_loss:.6f})")

                # Log test metrics to wandb with epoch number
                if wandb.run:
                    # Create a dictionary with epoch and all test metrics
                    epoch_metrics = {
                        "epoch": epoch + 1,
                        "test_f1": test_results['f1'],
                        "test_precision": test_results['precision'],
                        "test_recall": test_results['recall'],
                        "test_eTaP": test_results['eTaP'],
                        "test_eTaR": test_results['eTaR'],
                        "test_eTaF1": test_results['eTaF1'],
                        "test_scenario_detection_rate": test_results['scenario_detection_rate'],
                        "test_fp_alarms": test_results['fp_alarms']
                    }
                    wandb.log(epoch_metrics)
                    print(f"Logged epoch {epoch+1} metrics to wandb")

                if test_f1 - best_f1 > min_delta:
                    best_f1 = test_f1
                    epochs_no_improve = 0
                    torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))
                    print(f"*** New best F1: {best_f1:.4f}. Model saved. ***")
                else:
                    epochs_no_improve += 1
                    if early_stopping and epochs_no_improve >= patience:
                        print(f"Early stopping triggered after {patience} epochs.")
                        break
        
        print(f"\nTraining complete. Best F1 on test set: {best_f1:.4f}")
        # Load best model and return final evaluation
        best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            print("Loaded best model for final evaluation.")
        
        final_results = self.evaluate(self.test_dataloader, "Final Test")
        
        # Create detection timeline visualization
        self.create_detection_timeline_visualization()
        
        # Analyze individual attacks
        self.analyze_individual_attacks()
        
        return final_results 