import torch
from torch import nn

class ResidualDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super().__init__()
        self.n_layers = n_layers
        
        if n_layers == 1:
            # Single layer case (original behavior)
            self.layers = nn.Sequential(
                nn.Linear(input_dim, output_dim)
            )
        elif n_layers == 2:
            # Two layer case (current default)
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            # Multi-layer case (deeper networks)
            layers = []
            # First layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Add dropout for regularization
            
            # Hidden layers
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
            
            # Output layer
            layers.append(nn.Linear(hidden_dim, output_dim))
            
            self.layers = nn.Sequential(*layers)
        
        # Residual projection (needed when input_dim != output_dim)
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    
    def forward(self, x):
        # Residual connection: output = F(x) + projection(x)
        return self.layers(x) + self.shortcut(x)

class EnhancedResidualDecoder(nn.Module):
    """
    Enhanced residual decoder with multiple skip connections for better gradient flow.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Build layers with intermediate skip connections
        self.layer_modules = nn.ModuleList()
        
        # First layer
        self.layer_modules.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(n_layers - 2):
            self.layer_modules.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.layer_modules.append(nn.Linear(hidden_dim, output_dim))
        
        # Skip connection projections
        self.skip_projections = nn.ModuleList()
        for i in range(n_layers - 1):
            if i == 0:
                # First skip: input_dim -> hidden_dim
                self.skip_projections.append(nn.Linear(input_dim, hidden_dim))
            else:
                # Middle skips: hidden_dim -> hidden_dim
                self.skip_projections.append(nn.Identity())
        
        # Final skip: input_dim -> output_dim
        self.final_skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Store input for final skip connection
        x_input = x
        
        # Forward pass with intermediate skip connections
        for i, layer in enumerate(self.layer_modules[:-1]):
            # Apply layer
            x = layer(x)
            
            # Add skip connection if not the last hidden layer
            if i < len(self.skip_projections):
                skip = self.skip_projections[i](x_input if i == 0 else x)
                x = x + skip
            
            # Apply activation and dropout
            x = self.dropout(self.activation(x))
        
        # Final layer
        x = self.layer_modules[-1](x)
        
        # Final skip connection
        x = x + self.final_skip(x_input)
        
        return x

class MultiScaleDecoder(nn.Module):
    """
    Multi-scale decoder that can handle concatenated features from multiple layers.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super().__init__()
        self.n_layers = n_layers
        
        # Handle potentially larger input due to concatenation
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        layers = []
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        
        # Skip connection
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    
    def forward(self, x):
        x_proj = self.input_projection(x)
        x_out = self.layers(x_proj)
        return x_out + self.shortcut(x)
    
class TemporalConvNet(nn.Module):
    """
    Lightweight Temporal Convolutional Network for sequence modeling.
    Uses causal dilated convolutions to capture temporal patterns.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, kernel_size=3):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size, 
                                   dilation=1, padding=(kernel_size-1)))
        
        # Additional layers with increasing dilation
        for i in range(1, num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            self.layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                                       dilation=dilation, padding=padding))
        
        # Final projection
        self.final_conv = nn.Conv1d(hidden_dim, output_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim] 
        Returns: [batch_size, output_dim] (last timestep prediction)
        """
        # Transpose to [batch_size, input_dim, seq_len] for Conv1d
        x = x.transpose(1, 2)
        
        for layer in self.layers:
            x = self.dropout(self.relu(layer(x)))
            # Causal: remove future information
            x = x[:, :, :-layer.padding[0]] if layer.padding[0] > 0 else x
            
        x = self.final_conv(x)
        
        # Return last timestep: [batch_size, output_dim]
        return x[:, :, -1] 