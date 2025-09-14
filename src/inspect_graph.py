# src/inspect_graph.py - Enhanced for rich features and better graphs
import os
import glob
import random
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import numpy as np

def inspect_random_graph(graph_dir="data/graphs"):
    """Load and inspect a random PyTorch Geometric graph with rich features."""
    graph_files = glob.glob(os.path.join(graph_dir, "*.pt"))
    if not graph_files:
        print(f"⚠️ No graphs found in {graph_dir}")
        return None, None
    
    graph_file = random.choice(graph_files)
    print(f"\n📂 Loading graph: {graph_file}")
    
    # Load PyTorch Geometric Data object
    data = torch.load(graph_file, weights_only=False)
    print(f"✅ Graph loaded: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
    print(f"Graph label: {'Useful' if data.y.item() == 1 else 'Not Useful'}")
    print(f"Node features shape: {data.x.shape}")
    print(f"Edge index shape: {data.edge_index.shape}")
    
    # Check for edge attributes
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        print(f"Edge attributes shape: {data.edge_attr.shape}")
        print(f"Edge weights range: {data.edge_attr.min().item():.3f} - {data.edge_attr.max().item():.3f}")
    
    # Enhanced feature analysis for rich embeddings
    if data.x.shape[1] > 20:  # Rich features
        print(f"\n🧠 Rich Feature Analysis:")
        
        # Analyze token types (first feature should still be token type)
        if data.x.shape[1] > 0:
            token_types = data.x[:, 0]
            code_nodes = (token_types == 0).sum().item()
            comment_nodes = (token_types == 1).sum().item()
            print(f"  📊 Token distribution: {code_nodes} code, {comment_nodes} comment nodes")
        
        # Feature statistics
        print(f"  📈 Feature statistics:")
        print(f"    Mean: {data.x.mean(dim=0)[:10]}...")  # First 10 features
        print(f"    Std:  {data.x.std(dim=0)[:10]}...")
        print(f"    Min:  {data.x.min(dim=0)[0][:10]}...")
        print(f"    Max:  {data.x.max(dim=0)[0][:10]}...")
        
        # Check for embedding regions
        if data.x.shape[1] >= 128:
            word2vec_region = data.x[:, 17:17+128]  # Skip statistical features
            print(f"  🔤 Word2Vec region stats:")
            print(f"    Non-zero features: {(word2vec_region != 0).sum().item()}/{word2vec_region.numel()}")
            print(f"    Mean magnitude: {word2vec_region.abs().mean().item():.4f}")
    
    else:  # Simple features
        print(f"\n📊 Simple Feature Analysis:")
        if data.x.shape[1] > 0:
            token_types = data.x[:, 0]
            code_nodes = (token_types == 0).sum().item()
            comment_nodes = (token_types == 1).sum().item()
            print(f"  Token types: {code_nodes} code, {comment_nodes} comment")
    
    # Graph connectivity analysis
    print(f"\n🔗 Graph Connectivity:")
    edges = data.edge_index.t().numpy()
    
    # Analyze edge types
    edge_types = {"code-code": 0, "comment-comment": 0, "code-comment": 0, "self-loops": 0}
    
    for edge in edges:
        src, dst = edge[0], edge[1]
        if src == dst:
            edge_types["self-loops"] += 1
        elif data.x[src, 0] == 0 and data.x[dst, 0] == 0:  # Both code
            edge_types["code-code"] += 1
        elif data.x[src, 0] == 1 and data.x[dst, 0] == 1:  # Both comment
            edge_types["comment-comment"] += 1
        else:  # Mixed
            edge_types["code-comment"] += 1
    
    print(f"  Edge types: {edge_types}")
    
    # Show sample edges with weights if available
    print(f"\n🌉 Sample edges (first 10):")
    for i in range(min(10, data.edge_index.shape[1])):
        src, dst = data.edge_index[:, i].tolist()
        src_type = "code" if data.x[src, 0] == 0 else "comment"
        dst_type = "code" if data.x[dst, 0] == 0 else "comment"
        
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            weight = data.edge_attr[i].item()
            print(f"    {src}({src_type}) → {dst}({dst_type}) [weight: {weight:.2f}]")
        else:
            print(f"    {src}({src_type}) → {dst}({dst_type})")
    
    return data, graph_file

def visualize_enhanced_graph(data, title="Enhanced Graph Visualization", show_weights=True):
    """Visualize enhanced graph with better layout and edge weights."""
    if data is None:
        return
    
    # Convert to NetworkX
    G = nx.Graph()
    
    # Add nodes with rich information
    num_nodes = data.x.shape[0]
    for i in range(num_nodes):
        token_type = "code" if data.x[i, 0] == 0 else "comment"
        
        # Try to get more meaningful node info if available
        if data.x.shape[1] > 5:
            # Rich features - use first few statistical features
            node_info = {
                'type': token_type,
                'length': data.x[i, 1].item() if data.x.shape[1] > 1 else 0,
                'is_numeric': data.x[i, 2].item() if data.x.shape[1] > 2 else 0,
                'feature_norm': data.x[i].norm().item()
            }
        else:
            node_info = {'type': token_type}
            
        G.add_node(i, **node_info)
    
    # Add edges with weights
    edges = data.edge_index.t().tolist()
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        edge_weights = data.edge_attr.squeeze().tolist()
        if not isinstance(edge_weights, list):
            edge_weights = [edge_weights]
        G.add_weighted_edges_from([(u, v, w) for (u, v), w in zip(edges, edge_weights)])
    else:
        G.add_edges_from(edges)
    
    # Create visualization with better layout
    plt.figure(figsize=(16, 12))
    
    # Separate code and comment nodes
    code_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'code']
    comment_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'comment']
    
    # Use hierarchical layout to separate code and comment sections
    pos = {}
    
    # Position code nodes on the left
    if code_nodes:
        code_pos = nx.spring_layout(G.subgraph(code_nodes), center=(-1, 0), k=0.5)
        pos.update(code_pos)
    
    # Position comment nodes on the right  
    if comment_nodes:
        comment_pos = nx.spring_layout(G.subgraph(comment_nodes), center=(1, 0), k=0.5)
        pos.update(comment_pos)
    
    # If we couldn't separate, use default layout
    if not pos:
        pos = nx.spring_layout(G, seed=42, k=1.5, iterations=100)
    
    # Draw code nodes (blue)
    if code_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=code_nodes,
            node_color="lightblue",
            node_size=400,
            alpha=0.8,
            label=f"Code tokens ({len(code_nodes)})"
        )
    
    # Draw comment nodes (green)
    if comment_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=comment_nodes,
            node_color="lightgreen", 
            node_size=400,
            alpha=0.8,
            label=f"Comment tokens ({len(comment_nodes)})"
        )
    
    # Draw edges with varying thickness based on weights
    if show_weights and hasattr(data, 'edge_attr') and data.edge_attr is not None:
        weights = data.edge_attr.squeeze().numpy()
        
        # Normalize weights for visualization
        min_weight, max_weight = weights.min(), weights.max()
        if max_weight > min_weight:
            normalized_weights = 1 + 4 * (weights - min_weight) / (max_weight - min_weight)
        else:
            normalized_weights = np.ones_like(weights) * 2
            
        nx.draw_networkx_edges(
            G, pos, 
            width=normalized_weights,
            edge_color="gray", 
            alpha=0.6
        )
    else:
        nx.draw_networkx_edges(G, pos, edge_color="gray", width=1.5, alpha=0.6)
    
    # Add node labels (just IDs, since we have many nodes)
    if num_nodes <= 50:  # Only show labels for smaller graphs
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
    
    # Title with comprehensive info
    label_text = "Useful" if data.y.item() == 1 else "Not Useful"
    feature_dim = data.x.shape[1]
    
    plt.title(f"{title}\n"
             f"Label: {label_text} | Nodes: {num_nodes} | Edges: {len(edges)} | "
             f"Features: {feature_dim}D", fontsize=12)
    
    plt.legend()
    plt.axis('off')
    
    # Add feature info as text
    info_text = f"Code: {len(code_nodes)} nodes\nComment: {len(comment_nodes)} nodes"
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        info_text += f"\nEdge weights: {weights.min():.2f}-{weights.max():.2f}"
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def compare_graphs(graph_dir="data/graphs", num_samples=3):
    """Compare multiple graphs side by side."""
    graph_files = glob.glob(os.path.join(graph_dir, "*.pt"))
    if len(graph_files) < num_samples:
        num_samples = len(graph_files)
    
    sample_files = random.sample(graph_files, num_samples)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(6*num_samples, 6))
    if num_samples == 1:
        axes = [axes]
    
    for i, graph_file in enumerate(sample_files):
        data = torch.load(graph_file, weights_only=False)
        
        # Convert to NetworkX for plotting
        G = nx.Graph()
        for j in range(data.x.shape[0]):
            token_type = "code" if data.x[j, 0] == 0 else "comment"
            G.add_node(j, type=token_type)
        
        edges = data.edge_index.t().tolist()
        G.add_edges_from(edges)
        
        # Separate node types
        code_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'code']
        comment_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'comment']
        
        # Simple layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw on subplot
        axes[i].set_title(f"Graph {i+1}\n{'Useful' if data.y.item() == 1 else 'Not Useful'}")
        
        if code_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=code_nodes, node_color="lightblue", 
                                 node_size=100, alpha=0.8, ax=axes[i])
        if comment_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=comment_nodes, node_color="lightgreen", 
                                 node_size=100, alpha=0.8, ax=axes[i])
        
        nx.draw_networkx_edges(G, pos, edge_color="gray", width=0.5, alpha=0.6, ax=axes[i])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def inspect_multiple_graphs(graph_dir="data/graphs", num_samples=5):
    """Inspect multiple random graphs with detailed statistics."""
    graph_files = glob.glob(os.path.join(graph_dir, "*.pt"))
    if not graph_files:
        print(f"⚠️ No graphs found in {graph_dir}")
        return
    
    sample_files = random.sample(graph_files, min(num_samples, len(graph_files)))
    
    print(f"\n📊 Analyzing {len(sample_files)} random graphs...")
    print("=" * 80)
    
    stats = {
        'node_counts': [],
        'edge_counts': [],
        'feature_dims': [],
        'useful_count': 0,
        'code_node_counts': [],
        'comment_node_counts': []
    }
    
    for i, graph_file in enumerate(sample_files):
        data = torch.load(graph_file, weights_only=False)
        
        # Basic stats
        num_nodes = data.x.shape[0]
        num_edges = data.edge_index.shape[1]
        feature_dim = data.x.shape[1]
        label = data.y.item()
        
        # Token type counts
        code_count = (data.x[:, 0] == 0).sum().item()
        comment_count = (data.x[:, 0] == 1).sum().item()
        
        # Store stats
        stats['node_counts'].append(num_nodes)
        stats['edge_counts'].append(num_edges)
        stats['feature_dims'].append(feature_dim)
        stats['code_node_counts'].append(code_count)
        stats['comment_node_counts'].append(comment_count)
        if label == 1:
            stats['useful_count'] += 1
        
        # Print individual graph info
        print(f"Graph {i+1:2d}: {os.path.basename(graph_file)}")
        print(f"  📊 Nodes: {num_nodes:3d} | Edges: {num_edges:3d} | Features: {feature_dim}D")
        print(f"  🏷️  Label: {'Useful' if label == 1 else 'Not Useful'}")
        print(f"  🔤 Tokens: {code_count} code, {comment_count} comment")
        
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_weights = data.edge_attr.squeeze()
            print(f"  🔗 Edge weights: {edge_weights.min():.2f} - {edge_weights.max():.2f}")
        print()
    
    # Summary statistics
    print("📈 SUMMARY STATISTICS:")
    print(f"  Nodes: {np.mean(stats['node_counts']):.1f} ± {np.std(stats['node_counts']):.1f}")
    print(f"  Edges: {np.mean(stats['edge_counts']):.1f} ± {np.std(stats['edge_counts']):.1f}")
    print(f"  Features: {stats['feature_dims'][0]}D (consistent: {len(set(stats['feature_dims'])) == 1})")
    print(f"  Labels: {stats['useful_count']}/{len(sample_files)} useful ({100*stats['useful_count']/len(sample_files):.1f}%)")
    print(f"  Code tokens: {np.mean(stats['code_node_counts']):.1f} ± {np.std(stats['code_node_counts']):.1f}")
    print(f"  Comment tokens: {np.mean(stats['comment_node_counts']):.1f} ± {np.std(stats['comment_node_counts']):.1f}")

def main():
    print("🔍 Enhanced Graph Inspection Tool")
    print("=" * 50)
    
    # Detailed inspection of one graph
    data, graph_file = inspect_random_graph("data/graphs")
    if data is not None:
        print(f"\n🎨 Visualizing: {os.path.basename(graph_file)}")
        visualize_enhanced_graph(data, title=f"Enhanced Graph: {os.path.basename(graph_file)}")
    
    # Statistical overview of multiple graphs
    print("\n" + "=" * 50)
    inspect_multiple_graphs("data/graphs", num_samples=8)
    
    # Side-by-side comparison
    print("\n🔍 Comparing multiple graphs...")
    compare_graphs("data/graphs", num_samples=3)

if __name__ == "__main__":
    main()