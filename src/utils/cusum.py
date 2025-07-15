"""
CUSUM (Cumulative Sum) algorithm for robust anomaly detection.
This module provides a class to track cumulative deviations and detect
anomalies when the sum exceeds a dynamically calibrated threshold.
"""
import torch
import numpy as np

class CUSUM:
    """
    Implements the CUSUM change detection algorithm for multiple features.
    """
    def __init__(self, num_features, S=1.42, G=5.98, device='cpu'):
        """
        Initializes the CUSUM detector.

        Args:
            num_features (int): The number of features (sensors) to monitor.
            S (float): The scaling factor for the detection threshold.
            G (float): The growth factor to limit CUSUM score overshoot.
            device (str): The device to store tensors on ('cpu' or 'cuda').
        """
        self.num_features = num_features
        self.S = S
        self.G = G
        self.device = device
        
        # Initialize CUSUM scores, drift (delta), and thresholds (T)
        self.cusum_scores = torch.zeros(self.num_features, device=self.device)
        self.delta = torch.zeros(self.num_features, device=self.device)
        self.T = torch.zeros(self.num_features, device=self.device)
        
        self.is_calibrated = False

    def calibrate(self, residuals):
        """
        Calibrates the CUSUM parameters using residuals from normal data.

        Args:
            residuals (torch.Tensor): A 2D tensor of shape (num_samples, num_features)
                                      containing the absolute errors from normal data.
        """
        if residuals.shape[1] != self.num_features:
            raise ValueError(f"Input residuals shape ({residuals.shape[1]}) does not match num_features ({self.num_features})")
        
        print("Calibrating CUSUM parameters...")
        # Calculate drift (delta) as mean + 1*std of normal residuals
        self.delta = torch.mean(residuals, dim=0) + torch.std(residuals, dim=0)
        
        # Calculate CUSUM scores over the calibration data to find the max value
        temp_cusum_scores = torch.zeros_like(self.cusum_scores)
        max_cusum_so_far = torch.zeros_like(self.cusum_scores)

        for t in range(residuals.shape[0]):
            update_val = residuals[t] - self.delta
            temp_cusum_scores = torch.max(torch.zeros_like(temp_cusum_scores), temp_cusum_scores + update_val)
            max_cusum_so_far = torch.max(max_cusum_so_far, temp_cusum_scores)
            
        # Set the detection threshold T
        self.T = self.S * max_cusum_so_far
        self.is_calibrated = True
        
        print(f"CUSUM calibrated. Example delta: {self.delta[0]:.4f}, Example threshold T: {self.T[0]:.4f}")

    def update(self, current_residuals):
        """
        Update the CUSUM scores with a new residual and check for anomalies.

        Args:
            current_residuals (torch.Tensor): A 1D tensor of shape (num_features)
                                              with the absolute errors for the current timestep.

        Returns:
            torch.Tensor: A boolean tensor of shape (num_features) where True indicates
                          an anomaly for that feature.
        """
        if not self.is_calibrated:
            raise RuntimeError("CUSUM must be calibrated with normal data before updating.")
            
        # Update CUSUM score
        update_val = current_residuals - self.delta
        self.cusum_scores = torch.max(torch.zeros_like(self.cusum_scores), self.cusum_scores + update_val)

        # Apply growth factor limit to prevent massive overshooting
        growth_limit = self.T + self.G * self.delta
        self.cusum_scores = torch.min(self.cusum_scores, growth_limit)

        # Check for anomalies
        anomalies = self.cusum_scores > self.T
        
        return anomalies

    def reset(self):
        """Resets the CUSUM scores to zero."""
        self.cusum_scores = torch.zeros(self.num_features, device=self.device) 