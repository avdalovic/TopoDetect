import copy
import torch
from torch import nn
from topomodelx.nn.combinatorial.hmc import HMC

from src.models.common import ResidualDecoder, EnhancedResidualDecoder, MultiScaleDecoder, TemporalConvNet


class ResidualHMC(nn.Module):
    """
    HMC wrapper that adds residual connections for better gradient flow.
    """
    def __init__(self, channels_per_layer, negative_slope=0.2):
        super().__init__()
        self.hmc = HMC(
            channels_per_layer=channels_per_layer,
            negative_slope=negative_slope
        )
        self.use_residual = True
        print(f"Initialized ResidualHMC with {len(channels_per_layer)} layers")
    
    def forward(self, x_0, x_1, x_2, a0, a1, coa2, b1, b2):
        # Store initial inputs for residual connections
        x_0_init, x_1_init, x_2_init = x_0, x_1, x_2
        
        # Apply HMC
        h_0, h_1, h_2 = self.hmc(x_0, x_1, x_2, a0, a1, coa2, b1, b2)
        
        # Add residual connections if dimensions match
        if self.use_residual:
            if h_0.shape[-1] == x_0_init.shape[-1]:
                h_0 = h_0 + x_0_init
            if h_1.shape[-1] == x_1_init.shape[-1]:
                h_1 = h_1 + x_1_init
            if h_2.shape[-1] == x_2_init.shape[-1]:
                h_2 = h_2 + x_2_init
        
        return h_0, h_1, h_2


class AnomalyCCANN(nn.Module):
    """
    Anomaly detection model using Combinatorial Complex Attention Neural Network.
    Uses either an autoencoder approach to reconstruct features, or a temporal approach
    to predict the next timestep's features.
    """
    def __init__(self, channels_per_layer, original_feature_dims, temporal_mode=False, use_tcn=False, n_input=10, use_enhanced_decoders=False, use_categorical_embeddings=False, num_sensor_types=12, categorical_embedding_dim=8):
        """
        Initialize the anomaly detection model.
        
        Parameters
        ----------
        channels_per_layer : list
            Channel configuration for the HMC layers
        original_feature_dims : dict
            Dictionary containing feature dimensions for each cell type:
            {'0': dim_0, '1': dim_1, '2': dim_2}
        temporal_mode : bool, default=False
            Whether to use temporal mode
        use_tcn : bool, default=False
            Whether to use TCN instead of LSTM for temporal mode
        n_input : int, default=10
            Number of input timesteps for temporal mode
        use_enhanced_decoders : bool, default=False
            Whether to use enhanced decoders with skip connections
        """
        super().__init__()
        self.temporal_mode = temporal_mode
        self.use_tcn = use_tcn
        self.n_input = n_input
        self.original_feature_dims = original_feature_dims
        self.use_enhanced_decoders = use_enhanced_decoders
        self.use_categorical_embeddings = use_categorical_embeddings
        self.num_sensor_types = num_sensor_types
        self.categorical_embedding_dim = categorical_embedding_dim
        
        if self.temporal_mode:
            # In temporal mode, an LSTM/TCN first encodes the sequence.
            # The output of the temporal model becomes the input to the HMC encoder.
            self.lstm_hidden_dim = channels_per_layer[-1][-1][-1]  # Use final HMC channel size as LSTM hidden size
            
            # Create a new channel configuration for the HMC encoder to accept LSTM's output
            encoder_channels = copy.deepcopy(channels_per_layer)
            encoder_channels[0][0] = [self.lstm_hidden_dim] * 3  # HMC input must match LSTM hidden dim

            self.encoder = ResidualHMC(
                channels_per_layer=encoder_channels
            )
            print(f"Initialized AnomalyCCANN with LSTM->ResidualHMC architecture. LSTM hidden: {self.lstm_hidden_dim}")

            if self.use_tcn:
                # This path is not fully implemented for the new architecture yet
                print(f"Using TCN temporal mode with sequence length {n_input}")
                self.tcn_x0 = TemporalConvNet(input_dim=original_feature_dims['0'], hidden_dim=self.lstm_hidden_dim*2, output_dim=self.lstm_hidden_dim)
                self.tcn_x1 = TemporalConvNet(input_dim=original_feature_dims['1'], hidden_dim=self.lstm_hidden_dim*2, output_dim=self.lstm_hidden_dim)
                self.tcn_x2 = TemporalConvNet(input_dim=original_feature_dims['2'], hidden_dim=self.lstm_hidden_dim*2, output_dim=self.lstm_hidden_dim)
            else:
                print(f"Using LSTM temporal mode with sequence length {n_input}")
                # LSTMs to encode the original features over time - use appropriate input sizes
                self.lstm_encoder_x0 = nn.LSTM(input_size=original_feature_dims['0'], hidden_size=self.lstm_hidden_dim, batch_first=True)
                self.lstm_encoder_x1 = nn.LSTM(input_size=original_feature_dims['1'], hidden_size=self.lstm_hidden_dim, batch_first=True)
                self.lstm_encoder_x2 = nn.LSTM(input_size=original_feature_dims['2'], hidden_size=self.lstm_hidden_dim, batch_first=True)
            
            # Decoders map from HMC's output dimension back to original feature dimension
            decoder_input_dim = channels_per_layer[-1][-1][-1]
            
            # Create deeper decoders with more capacity for nonlinear reconstruction (temporal mode)
            decoder_hidden_dim = decoder_input_dim * 2  # 2x capacity for better reconstruction
            decoder_layers = 3  # Deeper decoders for complex patterns
            
            print(f"Creating deeper temporal decoders: input_dim={decoder_input_dim}, hidden_dim={decoder_hidden_dim}, layers={decoder_layers}")
            
            if self.use_enhanced_decoders:
                self.decoder_x0 = EnhancedResidualDecoder(input_dim=decoder_input_dim, hidden_dim=decoder_hidden_dim, output_dim=original_feature_dims['0'], n_layers=decoder_layers)
                self.decoder_x1 = EnhancedResidualDecoder(input_dim=decoder_input_dim, hidden_dim=decoder_hidden_dim, output_dim=original_feature_dims['1'], n_layers=decoder_layers)
                self.decoder_x2 = EnhancedResidualDecoder(input_dim=decoder_input_dim, hidden_dim=decoder_hidden_dim, output_dim=original_feature_dims['2'], n_layers=decoder_layers)
            else:
                self.decoder_x0 = ResidualDecoder(input_dim=decoder_input_dim, hidden_dim=decoder_hidden_dim, output_dim=original_feature_dims['0'], n_layers=decoder_layers)
                self.decoder_x1 = ResidualDecoder(input_dim=decoder_input_dim, hidden_dim=decoder_hidden_dim, output_dim=original_feature_dims['1'], n_layers=decoder_layers)
                self.decoder_x2 = ResidualDecoder(input_dim=decoder_input_dim, hidden_dim=decoder_hidden_dim, output_dim=original_feature_dims['2'], n_layers=decoder_layers)

        else: # Reconstruction mode
            print(f"Using reconstruction mode with enhanced decoders: {use_enhanced_decoders}")
            print(f"Feature dimensions: 0-cells={original_feature_dims['0']}, 1-cells={original_feature_dims['1']}, 2-cells={original_feature_dims['2']}")
            
            # Add categorical embedding layer if enabled
            if self.use_categorical_embeddings:
                self.categorical_embedding = nn.Embedding(num_sensor_types, categorical_embedding_dim)
                nn.init.xavier_uniform_(self.categorical_embedding.weight)
                print(f"Added categorical embedding layer: {num_sensor_types} types -> {categorical_embedding_dim}D")
                
                # Create enhanced channel configuration for HMC
                enhanced_channels = copy.deepcopy(channels_per_layer)
                enhanced_input_dim = original_feature_dims['0'] + categorical_embedding_dim
                enhanced_channels[0][0] = [enhanced_input_dim, channels_per_layer[0][0][1], channels_per_layer[0][0][2]]  # Only enhance 0-cells
                print(f"Enhanced HMC channels: {enhanced_channels}")
                print(f"Embeddings will enhance {original_feature_dims['0']}D features to {enhanced_input_dim}D for encoding")
                
                self.encoder = ResidualHMC(
                    channels_per_layer=enhanced_channels,
                    negative_slope = 0.2
                )
            else:
                self.encoder = ResidualHMC(
                    channels_per_layer=channels_per_layer,
                    negative_slope = 0.2
                )
            encoder_output_dim = channels_per_layer[-1][-1][-1]
            
            # Create deeper decoders with more capacity for nonlinear reconstruction
            decoder_hidden_dim = encoder_output_dim * 2  # 2x capacity for better reconstruction
            decoder_layers = 3  # Deeper decoders for complex patterns
            
            print(f"Creating deeper decoders: input_dim={encoder_output_dim}, hidden_dim={decoder_hidden_dim}, layers={decoder_layers}")
            
            if self.use_enhanced_decoders:
                print("Using EnhancedResidualDecoder with multiple skip connections")
                self.decoder_x0 = EnhancedResidualDecoder(input_dim=encoder_output_dim, hidden_dim=decoder_hidden_dim, output_dim=original_feature_dims['0'], n_layers=decoder_layers)
                self.decoder_x1 = EnhancedResidualDecoder(input_dim=encoder_output_dim, hidden_dim=decoder_hidden_dim, output_dim=original_feature_dims['1'], n_layers=decoder_layers)
                self.decoder_x2 = EnhancedResidualDecoder(input_dim=encoder_output_dim, hidden_dim=decoder_hidden_dim, output_dim=original_feature_dims['2'], n_layers=decoder_layers)
            else:
                print("Using standard ResidualDecoder")
                self.decoder_x0 = ResidualDecoder(input_dim=encoder_output_dim, hidden_dim=decoder_hidden_dim, output_dim=original_feature_dims['0'], n_layers=decoder_layers)
                self.decoder_x1 = ResidualDecoder(input_dim=encoder_output_dim, hidden_dim=decoder_hidden_dim, output_dim=original_feature_dims['1'], n_layers=decoder_layers)
                self.decoder_x2 = ResidualDecoder(input_dim=encoder_output_dim, hidden_dim=decoder_hidden_dim, output_dim=original_feature_dims['2'], n_layers=decoder_layers)

    def forward(self, *args, **kwargs):
        # Correctly dispatch based on the mode
        if self.temporal_mode:
            # In temporal mode, we expect a single dictionary argument.
            sample = args[0] if args else kwargs
            return self.forward_temporal(
                sample['x_0'], sample['x_1'], sample['x_2'],
                sample['a0'], sample['a1'], sample['coa2'],
                sample['b1'], sample['b2']
                )
        else:
            # In reconstruction mode, we expect positional arguments. Pass them through.
            return self.forward_original(*args, **kwargs)

    def forward_original(self, x_0_batch, x_1_batch, x_2_batch, a0, a1, coa2, b1, b2, type_ids=None):
        """
        Batched forward pass for reconstruction mode.
        This loops through the batch because the underlying topomodelx layers do not
        support batched tensor inputs.
        """
        batch_size = x_0_batch.shape[0]

        # Get static 2D topology matrices. If they have a batch dim, take the first item.
        a0_static = a0[0] if a0.dim() > 2 else a0
        a1_static = a1[0] if a1.dim() > 2 else a1
        coa2_static = coa2[0] if coa2.dim() > 2 else coa2
        b1_static = b1[0] if b1.dim() > 2 else b1
        b2_static = b2[0] if b2.dim() > 2 else b2

        recon_x0_list, recon_x1_list, recon_x2_list = [], [], []

        for i in range(batch_size):
            # Extract single sample features for this item in the batch
            x_0 = x_0_batch[i]
            x_1 = x_1_batch[i]
            x_2 = x_2_batch[i]

            # Store original features for loss computation (only the first 2 dimensions)
            x_0_original = x_0[:, :2]  # Only normalized_value + first_order_diff

            # Apply categorical embeddings if enabled
            if self.use_categorical_embeddings and type_ids is not None:
                # Get type IDs for this batch item
                batch_type_ids = type_ids[i] if type_ids.dim() > 1 else type_ids
                
                # Move type IDs to the same device as the model
                batch_type_ids = batch_type_ids.to(x_0.device)
                
                # Validate type IDs are within bounds (only print error if out of bounds)
                if batch_type_ids.max() >= self.num_sensor_types:
                    print(f"ERROR: Type ID {batch_type_ids.max().item()} exceeds embedding size {self.num_sensor_types}")
                    # Clamp to valid range
                    batch_type_ids = torch.clamp(batch_type_ids, 0, self.num_sensor_types - 1)
                    print(f"DEBUG: Clamped type IDs to range 0-{self.num_sensor_types - 1}")
                
                # Get categorical embeddings
                embeddings = self.categorical_embedding(batch_type_ids)  # (num_components, embedding_dim)
                
                # Create enhanced features by concatenating original features with embeddings
                # This way embeddings help encoding but aren't reconstructed
                x_0_enhanced = torch.cat([x_0_original, embeddings], dim=1)  # (num_components, 2 + embedding_dim)
                
                # Pass enhanced features to encoder
                h_0, h_1, h_2 = self.encoder(x_0_enhanced, x_1, x_2, a0_static, a1_static, coa2_static, b1_static, b2_static)
            else:
                # Standard encoding without embeddings
                h_0, h_1, h_2 = self.encoder(x_0, x_1, x_2, a0_static, a1_static, coa2_static, b1_static, b2_static)

            # Decode the results - ONLY reconstruct original features
            x0_reconstructed = self.decoder_x0(h_0)  # Shape: (num_components, 2) - only original features
            x1_reconstructed = self.decoder_x1(h_1)
            x2_reconstructed = self.decoder_x2(h_2)

            recon_x0_list.append(x0_reconstructed)
            recon_x1_list.append(x1_reconstructed)
            recon_x2_list.append(x2_reconstructed)
    
        # Stack the list of results back into a single batched tensor
        return torch.stack(recon_x0_list), torch.stack(recon_x1_list), torch.stack(recon_x2_list)
    
    def forward_temporal(self, seq_x0, seq_x1, seq_x2, a0, a1, coa2, b1, b2):
        """
        Forward pass for temporal prediction. New architecture: LSTM -> HMC -> Decoder.
        """
        batch_size = seq_x0.size(0)
        num_0_cells = seq_x0.size(2)
        num_1_cells = seq_x1.size(2)
        num_2_cells = seq_x2.size(2)

        # Reshape for LSTM: (batch * num_cells, seq_len, feat_dim)
        lstm_input_x0 = seq_x0.permute(0, 2, 1, 3).reshape(-1, self.n_input, seq_x0.shape[-1])
        lstm_input_x1 = seq_x1.permute(0, 2, 1, 3).reshape(-1, self.n_input, seq_x1.shape[-1])
        lstm_input_x2 = seq_x2.permute(0, 2, 1, 3).reshape(-1, self.n_input, seq_x2.shape[-1])
        
        # 1. Encode sequences with LSTM to get final hidden state
        _, (h_n_x0, _) = self.lstm_encoder_x0(lstm_input_x0)
        _, (h_n_x1, _) = self.lstm_encoder_x1(lstm_input_x1)
        _, (h_n_x2, _) = self.lstm_encoder_x2(lstm_input_x2)

        # Reshape LSTM output back to (batch, num_cells, hidden_dim)
        x_0_lstm_enc = h_n_x0.squeeze(0).reshape(batch_size, num_0_cells, -1)
        x_1_lstm_enc = h_n_x1.squeeze(0).reshape(batch_size, num_1_cells, -1)
        x_2_lstm_enc = h_n_x2.squeeze(0).reshape(batch_size, num_2_cells, -1)

        # 2. Pass LSTM-encoded features to HMC encoder
        x_0_hmc_enc, _, x_2_hmc_enc = self.encoder(
            x_0_lstm_enc, x_1_lstm_enc, x_2_lstm_enc, a0, a1, coa2, b1, b2
            )
            
        # 3. Decode HMC output to predict original features
        x0_pred = self.decoder_x0(x_0_hmc_enc)
        x2_pred = self.decoder_x2(x_2_hmc_enc)
        x2_mean_pred = torch.mean(x2_pred, dim=1, keepdim=True).squeeze()

        return x0_pred, x2_mean_pred
    
    def forward_tcn(self, seq_x0, seq_x1, seq_x2, a0, a1, coa2, b1, b2):
        # This method would also need to be updated to the new architecture
        # For now, it remains as a placeholder
        raise NotImplementedError("TCN forward pass is not updated to the new architecture yet.") 