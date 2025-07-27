#!/usr/bin/env python3
"""
Hybrid Topology Constructor for Topological Anomaly Detection

This module combines learned graph embeddings with existing domain knowledge
to create a hybrid topology that leverages both data-driven and expert knowledge.

Author: Research Student
"""

import numpy as np
import pickle
import torch
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

from src.models.graph_embedding_learner import GraphEmbeddingLearner
from src.utils.topology.swat_topology import SWATComplex
from src.utils.topology.wadi_topology import WADIComplex


class HybridTopologyConstructor:
    """
    Constructs hybrid topology combining learned embeddings with domain knowledge.
    
    This approach creates a three-level hierarchy:
    1. 0-cells: Individual components (sensors/actuators)  
    2. 1-cells: Relationships (learned + GECO + manual)
    3. 2-cells: Subsystems (PLC zones from domain knowledge)
    """
    
    def __init__(
        self,
        embeddings_file: str,
        dataset_type: str = 'swat',
        similarity_threshold: float = 0.1,
        use_geco: bool = True
    ):
        """
        Initialize hybrid topology constructor.
        
        Args:
            embeddings_file: Path to saved embeddings
            dataset_type: 'swat' or 'wadi'
            similarity_threshold: Minimum similarity for learned connections
            use_geco: Whether to include GECO relationships
        """
        self.embeddings_file = embeddings_file
        self.dataset_type = dataset_type.lower()
        self.similarity_threshold = similarity_threshold
        self.use_geco = use_geco
        
        # Load embeddings
        self.embedding_data = self._load_embeddings()
        self.node_names = self.embedding_data['node_names']
        self.similarity_matrix = self.embedding_data['similarity_matrix']
        self.top_k_data = self.embedding_data['top_k_neighbors']
        
        print(f"🔗 HybridTopologyConstructor initialized:")
        print(f"  📊 Dataset: {dataset_type.upper()}")
        print(f"  🧠 Embeddings: {len(self.node_names)} nodes")
        print(f"  🎯 Similarity threshold: {similarity_threshold}")
        print(f"  🤖 Use GECO: {use_geco}")
    
    def _load_embeddings(self) -> Dict:
        """Load embeddings from file."""
        with open(self.embeddings_file, 'rb') as f:
            return pickle.load(f)
    
    def get_learned_edges(self) -> List[Tuple[str, str, float]]:
        """
        Extract learned edges from embeddings based on similarity threshold.
        
        Returns:
            List of (source, target, similarity) tuples
        """
        learned_edges = []
        
        for i, node_name in enumerate(self.node_names):
            neighbor_indices = self.top_k_data['indices'][i]
            neighbor_similarities = self.top_k_data['similarities'][i]
            
            for neighbor_idx, similarity in zip(neighbor_indices, neighbor_similarities):
                if similarity > self.similarity_threshold:
                    neighbor_name = self.node_names[neighbor_idx]
                    # Avoid duplicate edges (only add if source < target lexicographically)
                    if node_name < neighbor_name:
                        learned_edges.append((node_name, neighbor_name, float(similarity)))
        
        return learned_edges
    
    def create_hybrid_complex(self):
        """
        Create hybrid combinatorial complex with learned + domain knowledge.
        
        Returns:
            Combinatorial complex with hybrid topology
        """
        print(f"\n🏗️ Building Hybrid Topology for {self.dataset_type.upper()}")
        print("=" * 60)
        
        # Get baseline topology from domain knowledge
        if self.dataset_type == 'swat':
            swat_builder = SWATComplex(use_geco_relationships=self.use_geco)
            base_complex = swat_builder.get_complex()
        elif self.dataset_type == 'wadi':
            wadi_builder = WADIComplex(use_geco_relationships=self.use_geco)
            base_complex = wadi_builder.get_complex()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
        
        # Get learned edges from embeddings
        learned_edges = self.get_learned_edges()
        
        print(f"📊 Topology Statistics:")
        print(f"  🏭 Baseline 2-cells (PLC zones): {len(list(base_complex.cells.hyperedge_dict[2]))}")
        print(f"  🔗 Baseline 1-cells (existing): {len(list(base_complex.cells.hyperedge_dict[1]))}")
        print(f"  🧠 Learned 1-cells (embeddings): {len(learned_edges)}")
        
        # Create hybrid adjacency matrix
        node_to_idx = {name: i for i, name in enumerate(self.node_names)}
        
        # Start with existing adjacency matrix (create if doesn't exist)
        try:
            # Try to get adjacency matrix from the complex
            adj_matrix = base_complex.adjacency_matrix(rank=0)
            hybrid_adjacency = adj_matrix.toarray().astype(float)
        except:
            # Create adjacency matrix based on complex structure
            num_nodes = len(self.node_names)
            hybrid_adjacency = np.zeros((num_nodes, num_nodes), dtype=float)
        
        # Add learned connections
        for source, target, similarity in learned_edges:
            if source in node_to_idx and target in node_to_idx:
                src_idx = node_to_idx[source]
                tgt_idx = node_to_idx[target]
                
                # Add bidirectional connection with similarity weight
                hybrid_adjacency[src_idx, tgt_idx] = similarity
                hybrid_adjacency[tgt_idx, src_idx] = similarity
        
        # Update the complex with hybrid adjacency
        # Note: TopoNetX combinatorial complex will be updated with new adjacency
        base_complex.hybrid_adjacency = hybrid_adjacency
        base_complex.learned_edges = learned_edges
        base_complex.embedding_metadata = {
            'embeddings_file': self.embeddings_file,
            'similarity_threshold': self.similarity_threshold,
            'num_learned_edges': len(learned_edges)
        }
        
        print(f"✅ Hybrid topology created!")
        print(f"  🔗 Total 1-cells: {np.count_nonzero(hybrid_adjacency) // 2}")
        print(f"  📈 Density increase: {(np.count_nonzero(hybrid_adjacency) / hybrid_adjacency.size) * 100:.2f}%")
        
        return base_complex
    
    def analyze_topology_changes(self, base_complex, hybrid_complex):
        """
        Analyze the differences between base and hybrid topology.
        
        Args:
            base_complex: Original topology
            hybrid_complex: Hybrid topology with learned edges
        """
        print(f"\n📈 Topology Analysis:")
        print("=" * 40)
        
        base_edges = len(list(base_complex.cells.hyperedge_dict[1]))
        learned_edges = len(self.get_learned_edges())
        
        print(f"🔗 Edge Statistics:")
        print(f"  Base edges (GECO + manual): {base_edges}")
        print(f"  Learned edges (embeddings): {learned_edges}")
        print(f"  Total hybrid edges: {base_edges + learned_edges}")
        print(f"  Increase factor: {(base_edges + learned_edges) / base_edges:.2f}x")
        
        # Analyze similarity distribution of learned edges
        similarities = [sim for _, _, sim in self.get_learned_edges()]
        if similarities:
            print(f"\n🎯 Learned Edge Similarities:")
            print(f"  Mean: {np.mean(similarities):.4f}")
            print(f"  Std:  {np.std(similarities):.4f}")
            print(f"  Min:  {np.min(similarities):.4f}")
            print(f"  Max:  {np.max(similarities):.4f}")
        
        # Analyze node degree changes  
        try:
            base_adj = base_complex.adjacency_matrix(rank=0).toarray()
        except:
            base_adj = np.zeros((len(self.node_names), len(self.node_names)))
        hybrid_adj = hybrid_complex.hybrid_adjacency
        
        base_degrees = np.sum(base_adj > 0, axis=1)
        hybrid_degrees = np.sum(hybrid_adj > 0, axis=1)
        degree_increases = hybrid_degrees - base_degrees
        
        print(f"\n📊 Node Degree Changes:")
        print(f"  Mean degree increase: {np.mean(degree_increases):.2f}")
        print(f"  Max degree increase:  {np.max(degree_increases)}")
        print(f"  Nodes with new edges: {np.sum(degree_increases > 0)}/{len(self.node_names)}")
    
    def print_top_learned_connections(self, num_connections: int = 10):
        """
        Print the top learned connections by similarity.
        
        Args:
            num_connections: Number of top connections to show
        """
        learned_edges = self.get_learned_edges()
        
        # Sort by similarity (descending)
        learned_edges.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\n🧠 Top {num_connections} Learned Connections:")
        print("=" * 60)
        
        for i, (source, target, similarity) in enumerate(learned_edges[:num_connections]):
            print(f"{i+1:2d}. {source:<12} ↔ {target:<12} (similarity: {similarity:.4f})")
    
    def save_hybrid_topology(self, output_file: str, hybrid_complex):
        """
        Save hybrid topology to file for reuse.
        
        Args:
            output_file: Path to save the hybrid topology
            hybrid_complex: The hybrid combinatorial complex
        """
        topology_data = {
            'dataset_type': self.dataset_type,
            'node_names': self.node_names,
            'learned_edges': self.get_learned_edges(),
            'similarity_threshold': self.similarity_threshold,
            'embedding_metadata': hybrid_complex.embedding_metadata,
            'hybrid_adjacency': hybrid_complex.hybrid_adjacency,
            'base_complex_info': {
                'num_0_cells': len(list(hybrid_complex.cells.hyperedge_dict[0])),
                'num_1_cells': len(list(hybrid_complex.cells.hyperedge_dict[1])),
                'num_2_cells': len(list(hybrid_complex.cells.hyperedge_dict[2]))
            }
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(topology_data, f)
        
        print(f"💾 Saved hybrid topology to: {output_file}")


def create_hybrid_swat_complex(
    embeddings_file: str,
    similarity_threshold: float = 0.1,
    use_geco: bool = True
):
    """
    Create hybrid SWAT combinatorial complex.
    
    Args:
        embeddings_file: Path to SWAT embeddings
        similarity_threshold: Minimum similarity for connections
        use_geco: Whether to include GECO relationships
        
    Returns:
        Hybrid SWAT combinatorial complex
    """
    constructor = HybridTopologyConstructor(
        embeddings_file=embeddings_file,
        dataset_type='swat',
        similarity_threshold=similarity_threshold,
        use_geco=use_geco
    )
    
    hybrid_complex = constructor.create_hybrid_complex()
    
    # Print analysis
    constructor.print_top_learned_connections(num_connections=10)
    
    return hybrid_complex


def create_hybrid_wadi_complex(
    embeddings_file: str,
    similarity_threshold: float = 0.1,
    use_geco: bool = True
):
    """
    Create hybrid WADI combinatorial complex.
    
    Args:
        embeddings_file: Path to WADI embeddings
        similarity_threshold: Minimum similarity for connections
        use_geco: Whether to include GECO relationships
        
    Returns:
        Hybrid WADI combinatorial complex
    """
    constructor = HybridTopologyConstructor(
        embeddings_file=embeddings_file,
        dataset_type='wadi',
        similarity_threshold=similarity_threshold,
        use_geco=use_geco
    )
    
    hybrid_complex = constructor.create_hybrid_complex()
    
    # Print analysis
    constructor.print_top_learned_connections(num_connections=10)
    
    return hybrid_complex


def test_hybrid_topology():
    """Test hybrid topology construction (for debugging)."""
    print("🧪 Testing Hybrid Topology Construction")
    print("=" * 50)
    
    # This is a placeholder - would need actual embedding files
    # embeddings_file = "embeddings/swat_embeddings.pkl"
    # hybrid_complex = create_hybrid_swat_complex(embeddings_file)
    
    print("⚠️  To test, run:")
    print("  1. python src/scripts/train_embeddings.py --config configs/geco_swat_flow_experiment.yaml")
    print("  2. Then use the generated embeddings with this module")


if __name__ == "__main__":
    test_hybrid_topology() 