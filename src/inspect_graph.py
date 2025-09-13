# src/inspect_graph.py
import os
import glob
import random
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

def inspect_random_graph(graph_dir="data/graphs"):
    """Load and inspect a random PyTorch Geometric graph."""
    graph_files = glob.glob(os.path.join(graph_dir, "*.pt"))
    if not graph_files:
        print(f"⚠️ No graphs found in {graph_dir}")
        return None, None
    
    graph_file = random.choice(graph_files)
    print(f"\n📂 Loading graph: {graph_file}")
    
    # Load PyTorch Geometric Data object with weights_only=False
    data = torch.load(graph_file, weights_only=False)
    print(f"✅ Graph loaded: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
    print(f"Graph label: {data.y.item() if data.y is not None else 'N/A'}")
    print(f"Node features shape: {data.x.shape}")
    print(f"Edge index shape: {data.edge_index.shape}")
    
    # Show feature statistics
    print(f"\nNode feature statistics:")
    print(f"  - Feature dimensions: {data.x.shape[1]}")
    print(f"  - Token type distribution (0=code, 1=comment):")
    if data.x.shape[1] > 0:
        token_types = data.x[:, 0]
        code_nodes = (token_types == 0).sum().item()
        comment_nodes = (token_types == 1).sum().item()
        print(f"    Code nodes: {code_nodes}, Comment nodes: {comment_nodes}")
    
    # Show some sample features
    print(f"\nSample node features (first 5 nodes):")
    for i in range(min(5, data.x.shape[0])):
        token_type = "code" if data.x[i, 0] == 0 else "comment"
        print(f"  Node {i} ({token_type}): {data.x[i].tolist()}")
    
    print(f"\nEdge connections (first 10 edges):")
    if data.edge_index.shape[1] > 0:
        for i in range(min(10, data.edge_index.shape[1])):
            src, dst = data.edge_index[:, i].tolist()
            src_type = "code" if data.x[src, 0] == 0 else "comment"
            dst_type = "code" if data.x[dst, 0] == 0 else "comment"
            print(f"  Edge {i}: {src}({src_type}) -> {dst}({dst_type})")
    
    return data, graph_file

def visualize_graph(data, title="Graph Visualization"):
    """Convert PyG Data to NetworkX and visualize with better colors."""
    if data is None:
        return
    
    # Convert PyTorch Geometric data to NetworkX
    G = nx.Graph()
    
    # Add nodes with token type information
    num_nodes = data.x.shape[0]
    for i in range(num_nodes):
        token_type = "code" if data.x[i, 0] == 0 else "comment"
        G.add_node(i, token_type=token_type)
    
    # Add edges
    edge_list = data.edge_index.t().tolist()
    G.add_edges_from(edge_list)
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Separate nodes by type for different colors
    code_nodes = [n for n, d in G.nodes(data=True) if d.get('token_type') == 'code']
    comment_nodes = [n for n, d in G.nodes(data=True) if d.get('token_type') == 'comment']
    
    # Use a layout that might separate code and comment sections
    if len(code_nodes) > 0 and len(comment_nodes) > 0:
        pos = nx.spring_layout(G, seed=42, k=2, iterations=100)
    else:
        pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
    
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
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=1.5, alpha=0.6)
    
    # Add node labels (node IDs)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
    
    plt.title(f"{title}\nNodes: {num_nodes}, Edges: {len(edge_list)}, Label: {data.y.item()}")
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def inspect_multiple_graphs(graph_dir="data/graphs", num_samples=3):
    """Inspect multiple random graphs."""
    graph_files = glob.glob(os.path.join(graph_dir, "*.pt"))
    if not graph_files:
        print(f"⚠️ No graphs found in {graph_dir}")
        return
    
    sample_files = random.sample(graph_files, min(num_samples, len(graph_files)))
    
    for graph_file in sample_files:
        print(f"\n{'='*50}")
        data = torch.load(graph_file, weights_only=False)
        print(f"📂 File: {os.path.basename(graph_file)}")
        print(f"📊 Nodes: {data.x.shape[0]}, Edges: {data.edge_index.shape[1]}")
        print(f"🏷️ Label: {data.y.item()}")
        print(f"📈 Node features: {data.x.shape}")
        
        # Token type breakdown
        if data.x.shape[1] > 0:
            code_count = (data.x[:, 0] == 0).sum().item()
            comment_count = (data.x[:, 0] == 1).sum().item()
            print(f"🔤 Tokens: {code_count} code, {comment_count} comment")

def main():
    print("🔍 Inspecting improved PyTorch Geometric graphs...")
    
    # Inspect one random graph in detail
    data, graph_file = inspect_random_graph("data/graphs")
    if data is not None:
        visualize_graph(data, title=f"Token Graph: {os.path.basename(graph_file)}")
    
    # Inspect multiple graphs (summary)
    print(f"\n{'='*50}")
    print("📊 Summary of multiple graphs:")
    inspect_multiple_graphs("data/graphs", num_samples=5)

if __name__ == "__main__":
    main()