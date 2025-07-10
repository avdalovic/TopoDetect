import os
import time
import pickle
import numpy as np
import torch
from torch import nn
import wandb

# New imports for CUSUM and advanced metrics
from src.utils.cusum import CUSUM
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
                 # Sampling parameters for metrics
                 sample_rate=1.0):
        """
        Initialize the trainer with additional training options.
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        
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
        print(f"Evaluation Method: {self.evaluation_method}")
        if self.evaluation_method == 'cusum':
            print(f"CUSUM Params: S={self.cusum_S}, G={self.cusum_G}")
        else:
            print(f"Threshold method: {threshold_method}")

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

        if self.evaluation_method == 'cusum':
            self.cusum_detector.reset()

        with torch.no_grad():
            for sample in dataloader:
                y_true = sample[-1].numpy() # Ground truth labels
                all_y_true.extend(y_true)

                if self.model.temporal_mode:
                    sample_device = self.to_device(sample)
                    x0_pred, _ = self.model(sample_device)
                    # Use L2 loss consistently
                    residuals_0 = self.crit(x0_pred, sample_device['target_x0'])
                    
                    # Get predictions for each item in the batch
                    batch_preds = []
                    for i in range(residuals_0.shape[0]):
                        if self.evaluation_method == 'cusum':
                            # Aggregate feature dimension before passing to CUSUM
                            residual_per_feature = torch.mean(residuals_0[i], dim=1)
                            sensor_anomalies = self.cusum_detector.update(residual_per_feature)
                            is_anomalous = torch.any(sensor_anomalies).item()
                            batch_preds.append(is_anomalous)
                        else: # Simple threshold
                            error = torch.mean(residuals_0[i])
                            is_anomalous = (error > self.anomaly_threshold_0).item()
                            batch_preds.append(is_anomalous)
                    all_y_pred.extend(batch_preds)
                    
                else: # Reconstruction - evaluate on all 3 levels
                    x_0, x_1, x_2, _, _, _, _, _, _ = self.to_device(sample)
                    x_0_recon, x_1_recon, x_2_recon = self.model(*self.to_device(sample)[:-1])
                    
                    # Use L2 loss consistently for all levels
                    residuals_0 = self.crit(x_0_recon, x_0)
                    residuals_1 = self.crit(x_1_recon, x_1)
                    residuals_2 = self.crit(x_2_recon, x_2)

                    # Get predictions for each item in the batch
                    batch_preds = []
                    for i in range(residuals_0.shape[0]):
                        if self.evaluation_method == 'cusum':
                            # For CUSUM, only use 0-cells
                            residual_per_feature = torch.mean(residuals_0[i], dim=1)
                            sensor_anomalies = self.cusum_detector.update(residual_per_feature)
                            is_anomalous = torch.any(sensor_anomalies).item()
                            batch_preds.append(is_anomalous)
                        else: # Multi-level threshold evaluation
                            error_0 = torch.mean(residuals_0[i])
                            error_1 = torch.mean(residuals_1[i])
                            error_2 = torch.mean(residuals_2[i])
                            
                            # Anomaly if ANY level exceeds its threshold
                            is_anomalous = (error_0 > self.anomaly_threshold_0).item() or \
                                         (error_1 > self.anomaly_threshold_1).item() or \
                                         (error_2 > self.anomaly_threshold_2).item()
                            batch_preds.append(is_anomalous)
                    all_y_pred.extend(batch_preds)

        # Convert lists to numpy arrays for metric calculations
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        
        # --- Calculate and Log All Metrics ---
        # Get attack segments and properly adjust for sampling
        attacks_full, _ = get_attack_indices("SWAT")
        
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
        fp_alarm_count = calculate_fp_alarms(all_y_true, all_y_pred, attacks, window_seconds=self.fp_alarm_window)
        time_aware_metrics = calculate_time_aware_metrics(all_y_true, all_y_pred, attacks)

        # Log results to console
        print("\n--- Evaluation Results ---")
        print(f"Standard Precision: {standard_metrics['precision']:.4f}")
        print(f"Standard Recall:    {standard_metrics['recall']:.4f}")
        print(f"Standard F1-Score:    {standard_metrics['f1']:.4f}")
        print("-" * 25)
        print(f"Scenario Detection Rate: {scenario_detection_rate:.4f} ({int(scenario_detection_rate*len(attacks))}/{len(attacks)})")
        print(f"False Positive Alarms:   {fp_alarm_count}")
        print("-" * 25)
        print(f"eTaP (Time-Aware Precision): {time_aware_metrics['eTaP']:.4f}")
        print(f"eTaR (Time-Aware Recall):    {time_aware_metrics['eTaR']:.4f}")
        print(f"eTaF1 (Time-Aware F1):       {time_aware_metrics['eTaF1']:.4f}")
        print("--------------------------\n")
        
        final_results = {**standard_metrics, **time_aware_metrics, 
                         'scenario_detection_rate': scenario_detection_rate,
                         'fp_alarms': fp_alarm_count}
        
        # Log to wandb if active
        if wandb.run:
            wandb.log(final_results)
            
        return final_results

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

            if (epoch + 1) % test_interval == 0:
                # Evaluate on test data
                test_results = self.evaluate(self.test_dataloader, "Test")
                test_f1 = test_results['f1']
                print(f"Epoch {epoch+1} Test F1: {test_f1:.4f} (Loss: {train_loss:.6f})")

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
        return self.evaluate(self.test_dataloader, "Final Test") 