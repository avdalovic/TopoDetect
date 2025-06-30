import torch
from torch import nn

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