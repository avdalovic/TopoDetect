#!/usr/bin/env python3
"""
Topology Comparison Script

Compare different topology approaches:
1. GECO + Embedding Hybrid
2. Pure Embedding (No GECO)
3. GECO Only (baseline)

This helps understand the contribution of each approach.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.topology.hybrid_topology import create_hybrid_swat_complex
from src.utils.topology.pure_embedding_topology import create_pure_embedding_swat_complex
from src.utils.topology.swat_topology import SWATComplex

def compare_topology_approaches():
    """Compare different topology construction approaches."""
    
    print("🔍 TOPOLOGY COMPARISON ANALYSIS")
    print("=" * 60)
    
    embeddings_file = "embeddings/swat_embeddings.pkl"
    similarity_threshold = 0.3
    
    # ========================================
    # 1. GECO Only (Baseline)
    # ========================================
    print("\n1️⃣ GECO ONLY (Baseline Approach)")
    print("-" * 40)
    
    geco_builder = SWATComplex(use_geco_relationships=True)
    geco_complex = geco_builder.get_complex()
    
    geco_edges = len(list(geco_complex.cells.hyperedge_dict[1]))
    geco_zones = len(list(geco_complex.cells.hyperedge_dict[2]))
    
    print(f"📊 GECO Structure:")
    print(f"  🔗 Edges (1-cells): {geco_edges}")
    print(f"  🏭 Zones (2-cells): {geco_zones}")
    print(f"  🎯 Approach: Domain knowledge from causal model")
    
    # ========================================
    # 2. Pure Embedding (No GECO)
    # ========================================
    print("\n2️⃣ PURE EMBEDDING (No GECO)")
    print("-" * 40)
    
    pure_complex = create_pure_embedding_swat_complex(
        embeddings_file=embeddings_file,
        similarity_threshold=similarity_threshold,
        min_edges_per_node=3
    )
    
    pure_edges = len(pure_complex.learned_edges)
    pure_zones = 5  # PLC zones
    
    print(f"\n📊 Pure Embedding vs GECO Baseline:")
    print(f"  🔗 Edges: {pure_edges} vs {geco_edges} ({pure_edges/geco_edges:.1f}x)")
    print(f"  🏭 Zones: {pure_zones} vs {geco_zones} (same)")
    print(f"  🎯 Approach: Pure data-driven discovery")
    
    # ========================================
    # 3. Hybrid (GECO + Embedding)
    # ========================================
    print("\n3️⃣ HYBRID (GECO + Embedding)")
    print("-" * 40)
    
    hybrid_complex = create_hybrid_swat_complex(
        embeddings_file=embeddings_file,
        similarity_threshold=similarity_threshold,
        use_geco=True
    )
    
    hybrid_edges = len(hybrid_complex.learned_edges) + geco_edges
    hybrid_zones = 5  # PLC zones
    
    print(f"\n📊 Hybrid vs GECO Baseline:")
    print(f"  🔗 Total edges: {hybrid_edges} vs {geco_edges} ({hybrid_edges/geco_edges:.1f}x)")
    print(f"  🧠 Learned edges: {len(hybrid_complex.learned_edges)}")
    print(f"  🤖 GECO edges: {geco_edges}")
    print(f"  🎯 Approach: Domain knowledge + data-driven discovery")
    
    # ========================================
    # 4. Analysis & Comparison
    # ========================================
    print("\n📈 COMPREHENSIVE ANALYSIS")
    print("=" * 50)
    
    approaches = {
        "GECO Only": {
            "edges": geco_edges,
            "learned_edges": 0,
            "coverage": "Domain knowledge",
            "complexity": "Low",
            "data_dependency": "None"
        },
        "Pure Embedding": {
            "edges": pure_edges,
            "learned_edges": pure_edges,
            "coverage": "Data-driven discovery",
            "complexity": "Medium",
            "data_dependency": "High"
        },
        "Hybrid": {
            "edges": hybrid_edges,
            "learned_edges": len(hybrid_complex.learned_edges),
            "coverage": "Domain + Data",
            "complexity": "High",
            "data_dependency": "Medium"
        }
    }
    
    print(f"{'Approach':<15} {'Edges':<8} {'Learned':<8} {'Coverage':<20} {'Complexity':<10}")
    print("-" * 70)
    for name, stats in approaches.items():
        print(f"{name:<15} {stats['edges']:<8} {stats['learned_edges']:<8} "
              f"{stats['coverage']:<20} {stats['complexity']:<10}")
    
    # ========================================
    # 5. Connection Analysis
    # ========================================
    print(f"\n🔗 CONNECTION OVERLAP ANALYSIS")
    print("-" * 35)
    
    # Get connection sets
    geco_connections = set()
    for i, cell in enumerate(geco_complex.cells.hyperedge_dict[1]):
        nodes = list(cell)
        if len(nodes) == 2:
            geco_connections.add(tuple(sorted(nodes)))
    
    pure_connections = set()
    for source, target, _ in pure_complex.learned_edges:
        pure_connections.add(tuple(sorted([source, target])))
    
    hybrid_learned_connections = set()
    for source, target, _ in hybrid_complex.learned_edges:
        hybrid_learned_connections.add(tuple(sorted([source, target])))
    
    # Calculate overlaps
    pure_geco_overlap = pure_connections.intersection(geco_connections)
    hybrid_geco_overlap = hybrid_learned_connections.intersection(geco_connections)
    pure_hybrid_overlap = pure_connections.intersection(hybrid_learned_connections)
    
    print(f"📊 Connection Overlaps:")
    print(f"  Pure ∩ GECO:        {len(pure_geco_overlap):<3} connections")
    print(f"  Hybrid ∩ GECO:      {len(hybrid_geco_overlap):<3} connections") 
    print(f"  Pure ∩ Hybrid:      {len(pure_hybrid_overlap):<3} connections")
    print(f"  Pure unique:        {len(pure_connections - geco_connections):<3} connections")
    print(f"  Hybrid unique:      {len(hybrid_learned_connections - geco_connections):<3} connections")
    
    # ========================================
    # 6. Top Connections Comparison
    # ========================================
    print(f"\n🧠 TOP LEARNED CONNECTIONS COMPARISON")
    print("-" * 45)
    
    # Pure embedding top connections
    pure_top = sorted(pure_complex.learned_edges, key=lambda x: x[2], reverse=True)[:5]
    print(f"🔍 Pure Embedding Top 5:")
    for i, (src, tgt, sim) in enumerate(pure_top):
        overlap_marker = "✓" if (src, tgt) in geco_connections or (tgt, src) in geco_connections else " "
        print(f"  {i+1}. {src:<10} ↔ {tgt:<10} ({sim:.3f}) {overlap_marker}")
    
    # Hybrid learned top connections
    hybrid_top = sorted(hybrid_complex.learned_edges, key=lambda x: x[2], reverse=True)[:5]
    print(f"\n🔀 Hybrid Learned Top 5:")
    for i, (src, tgt, sim) in enumerate(hybrid_top):
        overlap_marker = "✓" if (src, tgt) in geco_connections or (tgt, src) in geco_connections else " "
        print(f"  {i+1}. {src:<10} ↔ {tgt:<10} ({sim:.3f}) {overlap_marker}")
    
    print(f"\n✓ = Also found in GECO relationships")
    
    # ========================================
    # 7. Recommendations
    # ========================================
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 25)
    
    print(f"🎯 Use Pure Embedding when:")
    print(f"  • No domain knowledge available")
    print(f"  • Want to discover unknown relationships")
    print(f"  • GECO model may be inaccurate")
    print(f"  • Exploring new datasets")
    
    print(f"\n🎯 Use Hybrid when:")
    print(f"  • Have reliable domain knowledge")
    print(f"  • Want best of both worlds")
    print(f"  • Can afford higher complexity")
    print(f"  • Need maximum coverage")
    
    print(f"\n🎯 Use GECO Only when:")
    print(f"  • Domain knowledge is highly reliable")
    print(f"  • Want simple, interpretable model")
    print(f"  • Limited computational resources")
    print(f"  • Data quality is poor")
    
    return {
        'geco_complex': geco_complex,
        'pure_complex': pure_complex,
        'hybrid_complex': hybrid_complex,
        'analysis': approaches
    }


def main():
    """Main comparison function."""
    print("🚀 Starting Topology Comparison Analysis...")
    
    try:
        results = compare_topology_approaches()
        
        print(f"\n✅ Comparison completed successfully!")
        print(f"📊 Results available in returned dictionary")
        
        return results
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"💡 Make sure to run embedding training first:")
        print(f"   python src/scripts/train_embeddings.py --config configs/geco_swat_flow_experiment.yaml")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


if __name__ == "__main__":
    main() 