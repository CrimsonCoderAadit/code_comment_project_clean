# src/build_graph.py
import os
import ast
import pandas as pd
import networkx as nx

DATA_FILE = "data/Code_Comment_Seed_Data.csv"   # adjust if your CSV has a different name
OUT_DIR = "data/graphs"

def parse_code_to_ast_nodes(code, G, parent=None):
    """
    Recursively parse Python code AST and add nodes/edges to graph G.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return  # skip invalid code snippets

    def add_ast_node(node, parent_id=None):
        node_id = f"{type(node).__name__}_{id(node)}"
        G.add_node(node_id, type=type(node).__name__)
        if parent_id:
            G.add_edge(parent_id, node_id)

        for child in ast.iter_child_nodes(node):
            add_ast_node(child, node_id)

    add_ast_node(tree, parent)


def build_graph_from_row(row, idx):
    """
    Build a graph from one row of dataset.
    """
    G = nx.DiGraph()

    # Add comment node
    comment = str(row["Comments"]).strip()
    comment_node = f"comment_{idx}"
    G.add_node(comment_node, type="comment", text=comment)

    # Add AST nodes from code
    code = str(row["Surrounding Code Context"]).strip()
    parse_code_to_ast_nodes(code, G, parent=comment_node)

    # Add label at graph level
    label = str(row["Class"]).strip()
    G.graph["label"] = label

    return G


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    print(f"Columns in dataset: {df.columns.tolist()}")
    print(f"Building graphs for {len(df)} rows...")

    for idx, row in df.iterrows():
        G = build_graph_from_row(row, idx)
        out_file = os.path.join(OUT_DIR, f"graph_{idx}_label{G.graph['label']}.gpickle")
        nx.write_gpickle(G, out_file)

        if idx % 100 == 0:
            print(f"✅ Processed {idx} graphs...")

    print(f"\n🎉 Done! Graphs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
