#!/usr/bin/env python3
"""
Graph Embedding Learner for Topological Anomaly Detection

This module learns node embeddings during reconstruction tasks and uses cosine similarity
to determine graph topology, similar to Graph Deviation Network (GDN) approach.

Author: Research Student
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
import json
from pathlib import Path


class GraphEmbeddingLearner(nn.Module):
    """
    Learns node embeddings during reconstruction and computes similarity-based topology.
    
    This approach combines:
    1. Data-driven topology learning (like GDN)
    2. Domain knowledge (existing PLC zones)
    3. Efficient embedding reuse
    """
    
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int = 64,
        feature_dim: int = 2,
        top_k: int = 10,
        temperature: float = 1.0,
        device: str = 'cuda'
    ):
        """
        Initialize the Graph Embedding Learner.
        
        Args:
            num_nodes: Number of nodes (0-cells) in the graph
            embedding_dim: Dimension of learned embeddings
            feature_dim: Input feature dimension per node
            top_k: Number of top similar neighbors to connect
            temperature: Temperature for similarity computation
            device: Device to run on
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.temperature = temperature
        self.device = device
        
        # Node embedding layer - this will learn representations
        self.node_embeddings = nn.Parameter(
            torch.randn(num_nodes, embedding_dim, device=device) * 0.1
        )
        
        # Feature encoder to map input features to embedding space
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        ).to(device)
        
        # Reconstruction decoder
        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, feature_dim)
        ).to(device)
        
        print(f"🧠 GraphEmbeddingLearner initialized:")
        print(f"  📊 Nodes: {num_nodes}, Embedding dim: {embedding_dim}")
        print(f"  🔗 Top-{top_k} neighbors per node")
        print(f"  🎯 Feature dim: {feature_dim}")
        
    def forward(self, x: torch.Tensor, node_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode features, combine with embeddings, reconstruct.
        
        Args:
            x: Input features [batch_size, num_nodes, feature_dim]
            node_indices: Optional node indices for subset processing
            
        Returns:
            reconstructed_features: [batch_size, num_nodes, feature_dim]
            node_embeddings: [num_nodes, embedding_dim]
        """
        batch_size = x.shape[0]
        
        # Encode input features to embedding space
        encoded_features = self.feature_encoder(x)  # [batch_size, num_nodes, embedding_dim]
        
        # Combine with learned node embeddings
        combined_embeddings = encoded_features + self.node_embeddings.unsqueeze(0)  # Broadcasting
        
        # Reconstruct features from combined embeddings
        reconstructed = self.reconstruction_decoder(combined_embeddings)
        
        return reconstructed, self.node_embeddings
    
    def compute_similarity_matrix(self) -> torch.Tensor:
        """
        Compute cosine similarity matrix between all node embeddings.
        
        Returns:
            similarity_matrix: [num_nodes, num_nodes] cosine similarity matrix
        """
        # Normalize embeddings for cosine similarity
        normalized_embeddings = F.normalize(self.node_embeddings, p=2, dim=1)
        
        # Compute cosine similarity matrix
        similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
        
        # Apply temperature scaling
        similarity_matrix = similarity_matrix / self.temperature
        
        return similarity_matrix
    
    def get_top_k_neighbors(self, exclude_self: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k most similar neighbors for each node.
        
        Args:
            exclude_self: Whether to exclude self-connections
            
        Returns:
            neighbor_indices: [num_nodes, top_k] indices of top neighbors
            neighbor_similarities: [num_nodes, top_k] similarity scores
        """
        similarity_matrix = self.compute_similarity_matrix()
        
        if exclude_self:
            # Set diagonal to -inf to exclude self-connections
            similarity_matrix.fill_diagonal_(-float('inf'))
        
        # Get top-k neighbors for each node
        neighbor_similarities, neighbor_indices = torch.topk(
            similarity_matrix, k=self.top_k, dim=1, largest=True
        )
        
        return neighbor_indices, neighbor_similarities
    
    def get_learned_adjacency_matrix(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Create adjacency matrix from learned similarities.
        
        Args:
            threshold: Minimum similarity threshold for connections
            
        Returns:
            adjacency_matrix: [num_nodes, num_nodes] binary adjacency matrix
        """
        neighbor_indices, neighbor_similarities = self.get_top_k_neighbors()
        
        # Create adjacency matrix
        adjacency_matrix = torch.zeros(self.num_nodes, self.num_nodes, device=self.device)
        
        for node_idx in range(self.num_nodes):
            valid_neighbors = neighbor_similarities[node_idx] > threshold
            if valid_neighbors.any():
                neighbor_nodes = neighbor_indices[node_idx][valid_neighbors]
                adjacency_matrix[node_idx, neighbor_nodes] = 1
        
        # Make symmetric (if A->B then B->A)
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.t() > 0).float()
        
        return adjacency_matrix
    
    def save_embeddings(self, filepath: str, node_names: List[str], metadata: Dict = None):
        """
        Save learned embeddings and similarity information to file.
        
        Args:
            filepath: Path to save the embeddings
            node_names: List of node names corresponding to embeddings
            metadata: Additional metadata to save
        """
        # Compute final similarity matrix and top-k neighbors
        similarity_matrix = self.compute_similarity_matrix().detach().cpu().numpy()
        neighbor_indices, neighbor_similarities = self.get_top_k_neighbors()
        
        embedding_data = {
            'embeddings': self.node_embeddings.detach().cpu().numpy(),
            'similarity_matrix': similarity_matrix,
            'top_k_neighbors': {
                'indices': neighbor_indices.detach().cpu().numpy(),
                'similarities': neighbor_similarities.detach().cpu().numpy()
            },
            'node_names': node_names,
            'config': {
                'num_nodes': self.num_nodes,
                'embedding_dim': self.embedding_dim,
                'feature_dim': self.feature_dim,
                'top_k': self.top_k,
                'temperature': self.temperature
            },
            'metadata': metadata or {}
        }
        
        # Save as pickle file
        with open(filepath, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        print(f"💾 Saved embeddings to: {filepath}")
        print(f"  📊 {self.num_nodes} nodes, {self.embedding_dim}D embeddings")
        print(f"  🔗 Top-{self.top_k} neighbors per node")
        
        # Also save human-readable summary
        summary_path = filepath.replace('.pkl', '_summary.json')
        summary = {
            'num_nodes': self.num_nodes,
            'embedding_dim': self.embedding_dim,
            'top_k': self.top_k,
            'node_names': node_names,
            'similarity_stats': {
                'mean': float(np.mean(similarity_matrix)),
                'std': float(np.std(similarity_matrix)),
                'min': float(np.min(similarity_matrix)),
                'max': float(np.max(similarity_matrix))
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Saved summary to: {summary_path}")
    
    @classmethod
    def load_embeddings(cls, filepath: str, device: str = 'cuda') -> 'GraphEmbeddingLearner':
        """
        Load pre-trained embeddings from file.
        
        Args:
            filepath: Path to load embeddings from
            device: Device to load on
            
        Returns:
            Loaded GraphEmbeddingLearner instance
        """
        with open(filepath, 'rb') as f:
            embedding_data = pickle.load(f)
        
        config = embedding_data['config']
        learner = cls(
            num_nodes=config['num_nodes'],
            embedding_dim=config['embedding_dim'],
            feature_dim=config['feature_dim'],
            top_k=config['top_k'],
            temperature=config['temperature'],
            device=device
        )
        
        # Load the learned embeddings
        learner.node_embeddings.data = torch.tensor(
            embedding_data['embeddings'], device=device
        )
        
        print(f"📂 Loaded embeddings from: {filepath}")
        print(f"  📊 {config['num_nodes']} nodes, {config['embedding_dim']}D embeddings")
        
        return learner
    
    def print_top_neighbors(self, node_names: List[str], num_nodes_to_show: int = 5):
        """
        Print top neighbors for first few nodes (for debugging).
        
        Args:
            node_names: List of node names
            num_nodes_to_show: Number of nodes to show neighbors for
        """
        neighbor_indices, neighbor_similarities = self.get_top_k_neighbors()
        neighbor_indices = neighbor_indices.detach().cpu().numpy()
        neighbor_similarities = neighbor_similarities.detach().cpu().numpy()
        
        print(f"\n🔗 Top-{self.top_k} Neighbors (showing first {num_nodes_to_show} nodes):")
        print("=" * 60)
        
        for i in range(min(num_nodes_to_show, len(node_names))):
            node_name = node_names[i]
            neighbors = neighbor_indices[i]
            similarities = neighbor_similarities[i]
            
            print(f"\n📍 {node_name}:")
            for j, (neighbor_idx, similarity) in enumerate(zip(neighbors, similarities)):
                neighbor_name = node_names[neighbor_idx]
                print(f"  {j+1:2d}. {neighbor_name:<12} (similarity: {similarity:.4f})")


def create_embedding_based_topology(
    embeddings_file: str,
    existing_topology: Dict,
    similarity_threshold: float = 0.1
) -> Dict:
    """
    Create hybrid topology combining learned embeddings with existing domain knowledge.
    
    Args:
        embeddings_file: Path to saved embeddings
        existing_topology: Existing topology (PLC zones, etc.)
        similarity_threshold: Minimum similarity for connections
        
    Returns:
        hybrid_topology: Combined topology dictionary
    """
    # Load embeddings
    with open(embeddings_file, 'rb') as f:
        embedding_data = pickle.load(f)
    
    node_names = embedding_data['node_names']
    top_k_data = embedding_data['top_k_neighbors']
    similarity_matrix = embedding_data['similarity_matrix']
    
    # Create learned edges from similarity
    learned_edges = []
    for i, node_name in enumerate(node_names):
        neighbor_indices = top_k_data['indices'][i]
        neighbor_similarities = top_k_data['similarities'][i]
        
        for neighbor_idx, similarity in zip(neighbor_indices, neighbor_similarities):
            if similarity > similarity_threshold:
                neighbor_name = node_names[neighbor_idx]
                learned_edges.append({
                    'source': node_name,
                    'target': neighbor_name,
                    'similarity': float(similarity),
                    'type': 'learned'
                })
    
    # Combine with existing topology
    hybrid_topology = existing_topology.copy()
    hybrid_topology['learned_edges'] = learned_edges
    hybrid_topology['embedding_metadata'] = {
        'num_learned_edges': len(learned_edges),
        'similarity_threshold': similarity_threshold,
        'embeddings_file': embeddings_file
    }
    
    print(f"🔗 Created hybrid topology:")
    print(f"  📊 Learned edges: {len(learned_edges)}")
    print(f"  🏭 Existing 2-cells: {len(existing_topology.get('2_cells', []))}")
    
    return hybrid_topology 