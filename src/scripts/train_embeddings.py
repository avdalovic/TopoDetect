#!/usr/bin/env python3
"""
Embedding Training Script for Topological Anomaly Detection

This script learns node embeddings using reconstruction loss and saves them
for later use in topology construction.

Usage:
    python src/scripts/train_embeddings.py --config configs/geco_swat_flow_experiment.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.graph_embedding_learner import GraphEmbeddingLearner
from src.pipelines.swat_pipeline import load_swat_data
from src.pipelines.wadi_pipeline import load_wadi_data
from src.datasets.swat_dataset import SWaTDataset
from src.datasets.wadi_dataset import WADIDataset
from src.utils.topology.swat_topology import SWATComplex
from src.utils.topology.wadi_topology import WADIComplex


def train_embeddings_swat(config: dict, output_dir: str = "embeddings"):
    """
    Train embeddings for SWAT dataset.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save embeddings
    """
    print("🔥 Training SWAT Embeddings")
    print("=" * 50)
    
    # Load data
    train_path = os.path.join(config['data']['data_dir'], config['data']['train_path'])
    test_path = os.path.join(config['data']['data_dir'], config['data']['test_path'])
    
    train_data, validation_data, test_data = load_swat_data(
        train_path,
        test_path,
        sample_rate=config['data']['sample_rate'],
        validation_split_ratio=config['data']['validation_split_ratio']
    )
    
    # Create topology
    swat_builder = SWATComplex(use_geco_relationships=config['topology']['use_geco_relationships'])
    swat_complex = swat_builder.get_complex()
    
    # Create dataset (we only need the 0-cell features for embedding learning)
    dataset_args = {
        'temporal_mode': config['data']['temporal_mode'],
        'temporal_sample_rate': config['data']['temporal_sample_rate'],
        'use_geco_features': config['topology']['use_geco_features'],
        'normalization_method': config['data']['normalization_method'],
        'use_enhanced_2cell_features': config['data']['use_enhanced_2cell_features'],
        'use_first_order_differences': config['data']['use_first_order_differences'],
        'use_first_order_differences_edges': config['data']['use_first_order_differences_edges'],
        'use_flow_balance_features': config['data']['use_flow_balance_features'],
        'seed': config.get('seed', 42)
    }
    
    train_dataset = SWaTDataset(train_data, swat_builder, **dataset_args)
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True
    )
    
    # Get node information
    node_names = train_dataset.columns  # 0-cell node names
    num_nodes = len(node_names)
    feature_dim = train_dataset.feature_dim_0  # 0-cell feature dimension
    
    print(f"📊 Dataset Info:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Training samples: {len(train_dataset)}")
    
    # Initialize embedding learner
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_learner = GraphEmbeddingLearner(
        num_nodes=num_nodes,
        embedding_dim=config.get('embedding_dim', 64),
        feature_dim=feature_dim,
        top_k=config.get('top_k', 10),
        temperature=config.get('temperature', 1.0),
        device=device
    )
    
    # Training setup
    optimizer = torch.optim.Adam(embedding_learner.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()
    num_epochs = config.get('embedding_epochs', 50)
    
    print(f"\n🏋️ Training Setup:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Device: {device}")
    
    # Training loop
    embedding_learner.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in train_dataloader:
            # Get 0-cell features (x_0)
            x_0 = batch[0].to(device)  # [batch_size, num_nodes, feature_dim]
            
            # Forward pass
            x_0_recon, embeddings = embedding_learner(x_0)
            
            # Reconstruction loss
            loss = criterion(x_0_recon, x_0)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] Loss: {avg_loss:.6f}")
    
    print(f"\n✅ Training completed!")
    
    # Save embeddings
    os.makedirs(output_dir, exist_ok=True)
    embedding_file = os.path.join(output_dir, "swat_embeddings.pkl")
    
    metadata = {
        'dataset': 'SWAT',
        'sample_rate': config['data']['sample_rate'],
        'training_epochs': num_epochs,
        'final_loss': avg_loss,
        'config': config
    }
    
    embedding_learner.save_embeddings(embedding_file, node_names, metadata)
    
    # Print top neighbors for debugging
    embedding_learner.print_top_neighbors(node_names, num_nodes_to_show=3)
    
    return embedding_file


def train_embeddings_wadi(config: dict, output_dir: str = "embeddings"):
    """
    Train embeddings for WADI dataset.
    
    Args:
        config: Configuration dictionary  
        output_dir: Directory to save embeddings
    """
    print("🔥 Training WADI Embeddings")
    print("=" * 50)
    
    # Load data
    train_path = os.path.join(config['data']['data_dir'], config['data']['train_path'])
    test_path = os.path.join(config['data']['data_dir'], config['data']['test_path'])
    
    train_data, validation_data, test_data = load_wadi_data(
        train_path,
        test_path,
        sample_rate=config['data']['sample_rate'],
        validation_split_ratio=config['data']['validation_split_ratio']
    )
    
    # Create topology
    wadi_builder = WADIComplex(use_geco_relationships=config['topology']['use_geco_relationships'])
    wadi_complex = wadi_builder.get_complex()
    
    # Create dataset
    dataset_args = {
        'temporal_mode': config['data']['temporal_mode'],
        'temporal_sample_rate': config['data']['temporal_sample_rate'],
        'use_geco_features': config['topology']['use_geco_features'],
        'normalization_method': config['data']['normalization_method'],
        'use_enhanced_2cell_features': config['data']['use_enhanced_2cell_features'],
        'use_first_order_differences': config['data']['use_first_order_differences'],
        'use_first_order_differences_edges': config['data']['use_first_order_differences_edges'],
        'use_pressure_differential_features': config['data']['use_pressure_differential_features'],
        'seed': config.get('seed', 42)
    }
    
    train_dataset = WADIDataset(train_data, wadi_builder, **dataset_args)
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True
    )
    
    # Get node information
    node_names = train_dataset.columns
    num_nodes = len(node_names)
    feature_dim = train_dataset.feature_dim_0
    
    print(f"📊 Dataset Info:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Training samples: {len(train_dataset)}")
    
    # Initialize embedding learner
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_learner = GraphEmbeddingLearner(
        num_nodes=num_nodes,
        embedding_dim=config.get('embedding_dim', 64),
        feature_dim=feature_dim,
        top_k=config.get('top_k', 10),
        temperature=config.get('temperature', 1.0),
        device=device
    )
    
    # Training setup
    optimizer = torch.optim.Adam(embedding_learner.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()
    num_epochs = config.get('embedding_epochs', 50)
    
    print(f"\n🏋️ Training Setup:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Device: {device}")
    
    # Training loop
    embedding_learner.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in train_dataloader:
            # Get 0-cell features (x_0)
            x_0 = batch[0].to(device)
            
            # Forward pass
            x_0_recon, embeddings = embedding_learner(x_0)
            
            # Reconstruction loss
            loss = criterion(x_0_recon, x_0)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] Loss: {avg_loss:.6f}")
    
    print(f"\n✅ Training completed!")
    
    # Save embeddings
    os.makedirs(output_dir, exist_ok=True)
    embedding_file = os.path.join(output_dir, "wadi_embeddings.pkl")
    
    metadata = {
        'dataset': 'WADI',
        'sample_rate': config['data']['sample_rate'],
        'training_epochs': num_epochs,
        'final_loss': avg_loss,
        'config': config
    }
    
    embedding_learner.save_embeddings(embedding_file, node_names, metadata)
    
    # Print top neighbors for debugging
    embedding_learner.print_top_neighbors(node_names, num_nodes_to_show=3)
    
    return embedding_file


def main():
    parser = argparse.ArgumentParser(description='Train graph embeddings for topology learning')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--dataset', type=str, choices=['swat', 'wadi', 'auto'], default='auto',
                        help='Dataset to use (auto-detect from config if not specified)')
    parser.add_argument('--output_dir', type=str, default='embeddings', 
                        help='Directory to save embeddings')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top neighbors to connect')
    parser.add_argument('--embedding_epochs', type=int, default=50,
                        help='Number of epochs for embedding training')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for similarity computation')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    config['embedding_dim'] = args.embedding_dim
    config['top_k'] = args.top_k
    config['embedding_epochs'] = args.embedding_epochs
    config['temperature'] = args.temperature
    
    # Auto-detect dataset from config if not specified
    dataset = args.dataset
    if dataset == 'auto':
        if 'swat' in args.config.lower():
            dataset = 'swat'
        elif 'wadi' in args.config.lower():
            dataset = 'wadi'
        else:
            raise ValueError("Cannot auto-detect dataset. Please specify --dataset")
    
    print(f"🎯 Training embeddings for {dataset.upper()} dataset")
    print(f"📁 Config: {args.config}")
    print(f"💾 Output dir: {args.output_dir}")
    
    # Train embeddings
    if dataset == 'swat':
        embedding_file = train_embeddings_swat(config, args.output_dir)
    elif dataset == 'wadi':
        embedding_file = train_embeddings_wadi(config, args.output_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    print(f"\n🎉 Embedding training completed!")
    print(f"📄 Embeddings saved to: {embedding_file}")
    print(f"\n🔧 Next steps:")
    print(f"  1. Use embeddings in topology construction")
    print(f"  2. Run full training with hybrid topology")


if __name__ == "__main__":
    main() 