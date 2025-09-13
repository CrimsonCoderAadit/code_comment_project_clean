"""
Builds graph objects from raw CSVs and saves them as NetworkX gpickle files.
Each node has numeric features and each graph has a label (0 or 1).
"""

import os
import pandas as pd
import networkx as nx
from tqdm import tqdm

RAW_DATA = "data/raw/data.csv"
GRAPH_DIR = "data/graphs"

def ensure_dir(path):
    """Ensure directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)

def build_graph_from_row(code: str, comment: str, label: int, idx: int):
    """
    Build a toy graph:
    - Two nodes (code, comment)
    - Edge between them
    - Features: node length
    - Label: provided from dataset
    """
    G = nx.Graph()
    try:
        G.add_node(0, x=[len(code)], type="code")
        G.add_node(1, x=[len(comment)], type="comment")
        G.add_edge(0, 1)

        # Graph-level label
        G.graph["y"] = int(label)
        return G
    except Exception as e:
        print(f"⚠️ Skipping row {idx}: {e}")
        return None

def main():
    ensure_dir(GRAPH_DIR)
    df = pd.read_csv(RAW_DATA)

    print(f"Processing {len(df)} examples...")
    saved = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        code = str(row.get("code", ""))
        comment = str(row.get("comment", ""))
        label = int(row.get("label", 0))

        G = build_graph_from_row(code, comment, label, idx)
        if G is not None:
            out_file = os.path.join(GRAPH_DIR, f"graph_{idx}.gpickle")
            nx.write_gpickle(G, out_file)
            saved += 1
    print(f"✅ Saved {saved} graphs to {GRAPH_DIR}")

if __name__ == "__main__":
    main()
