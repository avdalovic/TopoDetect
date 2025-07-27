#!/usr/bin/env python3
"""
Pure Embedding-Based Topology Constructor

This module creates topology using ONLY learned embeddings without any GECO
or manual relationships. This is the purest data-driven approach similar to GDN.

Author: Research Student
"""

import numpy as np
import pickle
import torch
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import toponetx as tnx

from src.models.graph_embedding_learner import GraphEmbeddingLearner


class PureEmbeddingTopologyConstructor:
    """
    Creates topology using ONLY learned embeddings - no GECO, pure data-driven.
    
    This approach creates:
    1. 0-cells: Individual components (sensors/actuators)  
    2. 1-cells: ONLY learned relationships from embeddings
    3. 2-cells: PLC zones (domain knowledge for industrial process structure)
    """
    
    def __init__(
        self,
        embeddings_file: str,
        dataset_type: str = 'swat',
        similarity_threshold: float = 0.1,
        min_edges_per_node: int = 2
    ):
        """
        Initialize pure embedding topology constructor.
        
        Args:
            embeddings_file: Path to saved embeddings
            dataset_type: 'swat' or 'wadi'
            similarity_threshold: Minimum similarity for connections
            min_edges_per_node: Minimum edges per node to ensure connectivity
        """
        self.embeddings_file = embeddings_file
        self.dataset_type = dataset_type.lower()
        self.similarity_threshold = similarity_threshold
        self.min_edges_per_node = min_edges_per_node
        
        # Load embeddings
        self.embedding_data = self._load_embeddings()
        self.node_names = self.embedding_data['node_names']
        self.similarity_matrix = self.embedding_data['similarity_matrix']
        self.top_k_data = self.embedding_data['top_k_neighbors']
        
        print(f"🧠 PureEmbeddingTopologyConstructor initialized:")
        print(f"  📊 Dataset: {dataset_type.upper()}")
        print(f"  🔍 Embeddings: {len(self.node_names)} nodes")
        print(f"  🎯 Similarity threshold: {similarity_threshold}")
        print(f"  🔗 Min edges per node: {min_edges_per_node}")
        print(f"  🚫 NO GECO - Pure data-driven approach!")
    
    def _load_embeddings(self) -> Dict:
        """Load embeddings from file."""
        with open(self.embeddings_file, 'rb') as f:
            return pickle.load(f)
    
    def get_learned_edges(self) -> List[Tuple[str, str, float]]:
        """
        Extract learned edges from embeddings with adaptive thresholding.
        
        Returns:
            List of (source, target, similarity) tuples
        """
        learned_edges = []
        node_edge_count = defaultdict(int)
        
        # First pass: Add edges above threshold
        for i, node_name in enumerate(self.node_names):
            neighbor_indices = self.top_k_data['indices'][i]
            neighbor_similarities = self.top_k_data['similarities'][i]
            
            for neighbor_idx, similarity in zip(neighbor_indices, neighbor_similarities):
                if similarity > self.similarity_threshold:
                    neighbor_name = self.node_names[neighbor_idx]
                    # Avoid duplicate edges (only add if source < target lexicographically)
                    if node_name < neighbor_name:
                        learned_edges.append((node_name, neighbor_name, float(similarity)))
                        node_edge_count[node_name] += 1
                        node_edge_count[neighbor_name] += 1
        
        # Second pass: Ensure minimum connectivity for isolated nodes
        for i, node_name in enumerate(self.node_names):
            if node_edge_count[node_name] < self.min_edges_per_node:
                neighbor_indices = self.top_k_data['indices'][i]
                neighbor_similarities = self.top_k_data['similarities'][i]
                
                edges_needed = self.min_edges_per_node - node_edge_count[node_name]
                added_edges = 0
                
                for neighbor_idx, similarity in zip(neighbor_indices, neighbor_similarities):
                    if added_edges >= edges_needed:
                        break
                        
                    neighbor_name = self.node_names[neighbor_idx]
                    
                    # Check if this edge doesn't already exist
                    edge_exists = any(
                        (node_name == src and neighbor_name == tgt) or 
                        (node_name == tgt and neighbor_name == src)
                        for src, tgt, _ in learned_edges
                    )
                    
                    if not edge_exists and node_name != neighbor_name:
                        if node_name < neighbor_name:
                            learned_edges.append((node_name, neighbor_name, float(similarity)))
                        else:
                            learned_edges.append((neighbor_name, node_name, float(similarity)))
                        
                        node_edge_count[node_name] += 1
                        node_edge_count[neighbor_name] += 1
                        added_edges += 1
        
        return learned_edges
    
    def _get_plc_zones(self) -> List[Dict]:
        """
        Get PLC zone definitions (domain knowledge for process structure).
        
        Returns:
            List of PLC zone dictionaries
        """
        if self.dataset_type == 'swat':
            return [
                {
                    'name': 'PLC_1',
                    'components': ['FIT101', 'LIT101', 'MV101', 'P101']
                },
                {
                    'name': 'PLC_2', 
                    'components': ['AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 
                                  'P201', 'P202', 'P203', 'P204', 'P205', 'P206']
                },
                {
                    'name': 'PLC_3',
                    'components': ['MV201', 'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'P301']
                },
                {
                    'name': 'PLC_4',
                    'components': ['AIT402', 'FIT401', 'LIT401', 'P401']
                },
                {
                    'name': 'PLC_5',
                    'components': ['AIT501', 'FIT501', 'FIT502', 'P501']
                }
            ]
        elif self.dataset_type == 'wadi':
            return [
                {
                    'name': 'RawWater',
                    'components': ['1_AIT_001_PV', '1_AIT_002_PV', '1_FIT_001_PV', '1_LIT_001_PV']
                },
                {
                    'name': 'Elevated',
                    'components': ['1_LS_001_AL', '1_LS_002_AL', '1_MV_001_STATUS', '1_P_001_STATUS', '1_P_002_STATUS']
                },
                {
                    'name': 'Booster',
                    'components': ['2_DPIT_001_PV', '2_FIT_001_PV', '2_MV_001_STATUS', '2_P_001_STATUS', '2_P_002_STATUS']
                },
                {
                    'name': 'Consumers',
                    'components': ['2_FIC_101_CO', '2_FIC_101_PV', '2_FIC_101_SP', '2_FIC_201_CO', '2_FIC_201_PV', '2_FIC_201_SP']
                },
                {
                    'name': 'Return',
                    'components': ['3_AIT_001_PV', '3_AIT_002_PV', '3_FIT_001_PV', '3_LIT_001_PV']
                }
            ]
        else:
            return []
    
    def create_pure_embedding_complex(self):
        """
        Create combinatorial complex using ONLY learned embeddings.
        
        Returns:
            Combinatorial complex with pure data-driven topology
        """
        print(f"\n🧠 Building Pure Embedding Topology for {self.dataset_type.upper()}")
        print("=" * 70)
        print("🚫 NO GECO relationships - Pure data-driven approach!")
        
        # Create empty combinatorial complex
        complex = tnx.CombinatorialComplex()
        
        # Add 0-cells (nodes/components)
        for node_name in self.node_names:
            complex.add_node(node_name)
        
        print(f"📊 Added {len(self.node_names)} components as 0-cells")
        
        # Get learned edges from embeddings
        learned_edges = self.get_learned_edges()
        
        print(f"🧠 Learned edges from embeddings: {len(learned_edges)}")
        
        # Add 1-cells (learned relationships only)
        for source, target, similarity in learned_edges:
            if source in self.node_names and target in self.node_names:
                edge_name = f"learned_{source}_{target}"
                complex.add_cell([source, target], rank=1, name=edge_name)
        
        print(f"🔗 Added {len(learned_edges)} learned relationships as 1-cells")
        
        # Add 2-cells (PLC zones for process structure)
        plc_zones = self._get_plc_zones()
        added_zones = 0
        
        for zone in plc_zones:
            # Filter components that exist in our dataset
            existing_components = [comp for comp in zone['components'] if comp in self.node_names]
            
            if len(existing_components) >= 2:  # Need at least 2 components for a 2-cell
                complex.add_cell(existing_components, rank=2, name=zone['name'])
                added_zones += 1
                print(f"  Added 2-cell: {zone['name']} with {len(existing_components)} components")
        
        print(f"🏭 Added {added_zones} PLC zones as 2-cells")
        
        # Create adjacency matrix for learned connections
        node_to_idx = {name: i for i, name in enumerate(self.node_names)}
        num_nodes = len(self.node_names)
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
        
        for source, target, similarity in learned_edges:
            if source in node_to_idx and target in node_to_idx:
                src_idx = node_to_idx[source]
                tgt_idx = node_to_idx[target]
                
                # Add bidirectional connection with similarity weight
                adjacency_matrix[src_idx, tgt_idx] = similarity
                adjacency_matrix[tgt_idx, src_idx] = similarity
        
        # Attach metadata to complex
        complex.pure_embedding_adjacency = adjacency_matrix
        complex.learned_edges = learned_edges
        complex.embedding_metadata = {
            'embeddings_file': self.embeddings_file,
            'similarity_threshold': self.similarity_threshold,
            'num_learned_edges': len(learned_edges),
            'approach': 'pure_embedding_no_geco'
        }
        
        print(f"\n✅ Pure embedding topology created!")
        print(f"  📊 Structure: {len(self.node_names)} nodes, {len(learned_edges)} edges, {added_zones} zones")
        print(f"  🔗 Graph density: {(np.count_nonzero(adjacency_matrix) / adjacency_matrix.size) * 100:.2f}%")
        print(f"  🧠 Average node degree: {np.count_nonzero(adjacency_matrix) / len(self.node_names):.1f}")
        
        return complex
    
    def analyze_learned_topology(self, complex):
        """
        Analyze the learned topology structure.
        
        Args:
            complex: The pure embedding combinatorial complex
        """
        print(f"\n📈 Pure Embedding Topology Analysis:")
        print("=" * 50)
        
        learned_edges = self.get_learned_edges()
        
        # Analyze similarity distribution
        similarities = [sim for _, _, sim in learned_edges]
        if similarities:
            print(f"🎯 Learned Edge Similarities:")
            print(f"  Count: {len(similarities)}")
            print(f"  Mean:  {np.mean(similarities):.4f}")
            print(f"  Std:   {np.std(similarities):.4f}")
            print(f"  Min:   {np.min(similarities):.4f}")
            print(f"  Max:   {np.max(similarities):.4f}")
        
        # Analyze node connectivity
        adjacency = complex.pure_embedding_adjacency
        node_degrees = np.sum(adjacency > 0, axis=1)
        
        print(f"\n📊 Node Connectivity:")
        print(f"  Mean degree: {np.mean(node_degrees):.2f}")
        print(f"  Max degree:  {np.max(node_degrees)}")
        print(f"  Min degree:  {np.min(node_degrees)}")
        print(f"  Isolated nodes: {np.sum(node_degrees == 0)}")
        
        # Show most connected nodes
        top_connected_indices = np.argsort(node_degrees)[-5:][::-1]
        print(f"\n🔗 Most Connected Nodes:")
        for i, idx in enumerate(top_connected_indices):
            node_name = self.node_names[idx]
            degree = node_degrees[idx]
            print(f"  {i+1}. {node_name:<12} (degree: {degree})")
    
    def print_top_learned_connections(self, num_connections: int = 15):
        """
        Print the top learned connections by similarity.
        
        Args:
            num_connections: Number of top connections to show
        """
        learned_edges = self.get_learned_edges()
        
        # Sort by similarity (descending)
        learned_edges.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\n🧠 Top {num_connections} Learned Connections (Pure Embedding):")
        print("=" * 70)
        
        for i, (source, target, similarity) in enumerate(learned_edges[:num_connections]):
            print(f"{i+1:2d}. {source:<12} ↔ {target:<12} (similarity: {similarity:.4f})")


def create_pure_embedding_swat_complex(
    embeddings_file: str,
    similarity_threshold: float = 0.2,
    min_edges_per_node: int = 2
):
    """
    Create pure embedding SWAT combinatorial complex (NO GECO).
    
    Args:
        embeddings_file: Path to SWAT embeddings
        similarity_threshold: Minimum similarity for connections
        min_edges_per_node: Minimum edges per node for connectivity
        
    Returns:
        Pure embedding SWAT combinatorial complex
    """
    constructor = PureEmbeddingTopologyConstructor(
        embeddings_file=embeddings_file,
        dataset_type='swat',
        similarity_threshold=similarity_threshold,
        min_edges_per_node=min_edges_per_node
    )
    
    pure_complex = constructor.create_pure_embedding_complex()
    
    # Print analysis and top connections
    constructor.analyze_learned_topology(pure_complex)
    constructor.print_top_learned_connections(num_connections=15)
    
    return pure_complex


def create_pure_embedding_wadi_complex(
    embeddings_file: str,
    similarity_threshold: float = 0.2,
    min_edges_per_node: int = 2
):
    """
    Create pure embedding WADI combinatorial complex (NO GECO).
    
    Args:
        embeddings_file: Path to WADI embeddings
        similarity_threshold: Minimum similarity for connections
        min_edges_per_node: Minimum edges per node for connectivity
        
    Returns:
        Pure embedding WADI combinatorial complex
    """
    constructor = PureEmbeddingTopologyConstructor(
        embeddings_file=embeddings_file,
        dataset_type='wadi',
        similarity_threshold=similarity_threshold,
        min_edges_per_node=min_edges_per_node
    )
    
    pure_complex = constructor.create_pure_embedding_complex()
    
    # Print analysis and top connections
    constructor.analyze_learned_topology(pure_complex)
    constructor.print_top_learned_connections(num_connections=15)
    
    return pure_complex


def test_pure_embedding_topology():
    """Test pure embedding topology construction."""
    print("🧪 Testing Pure Embedding Topology Construction")
    print("=" * 55)
    
    print("⚠️  To test, run:")
    print("  1. python src/scripts/train_embeddings.py --config configs/geco_swat_flow_experiment.yaml")
    print("  2. Then use the generated embeddings with this module")
    print("  3. Compare with GECO+embedding hybrid approach!")


if __name__ == "__main__":
    test_pure_embedding_topology() 