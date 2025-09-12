# src/inspect_graph.py
import os
import glob
import random
import networkx as nx
import matplotlib.pyplot as plt


def inspect_random_graph(graph_dir="data/graphs"):
    graph_files = glob.glob(os.path.join(graph_dir, "*.gpickle"))
    if not graph_files:
        print(f"⚠️ No graphs found in {graph_dir}")
        return None

    graph_file = random.choice(graph_files)
    print(f"\n📂 Loading graph: {graph_file}")

    G = nx.read_gpickle(graph_file)
    print(f"✅ Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Graph label: {G.graph.get('label', 'N/A')}")

    # Show some debug info
    print("\nSample nodes with attributes:")
    for i, (node, data) in enumerate(G.nodes(data=True)):
        if i < 10:
            print(f"  {node}: {data}")

    return G, graph_file


def visualize_graph(G, title="Graph Visualization"):
    if G is None:
        return

    # Separate nodes by type
    comment_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "comment"]
    ast_nodes = [n for n, d in G.nodes(data=True) if d.get("type") != "comment"]

    # Layout: comment nodes on one side, AST nodes on the other
    pos = nx.spring_layout(G, seed=42, k=0.5)  # force-directed, looks cleaner

    plt.figure(figsize=(14, 10))

    # Draw AST nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=ast_nodes,
        node_color="skyblue",
        node_size=300,
        alpha=0.9,
        label="AST Nodes"
    )

    # Draw comment nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=comment_nodes,
        node_color="lightgreen",
        node_size=600,
        alpha=0.95,
        label="Comment Nodes"
    )

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=1.2, arrows=True)

    # Labels (truncate long ones)
    labels = {
        n: (d.get("text", d.get("type", str(n))))[:15] + "..."
        if len(str(d.get("text", d.get("type", str(n))))) > 15
        else d.get("text", d.get("type", str(n)))
        for n, d in G.nodes(data=True)
    }

    nx.draw_networkx_labels(G, pos, labels, font_size=6)

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    G, graph_file = inspect_random_graph("data/graphs")
    if G:
        visualize_graph(G, title=f"Visualization of {os.path.basename(graph_file)}")


if __name__ == "__main__":
    main()
