import os
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# ------------------------------
# Safe label conversion
# ------------------------------
def safe_label(label):
    """Convert labels to integers safely."""
    if pd.isna(label):
        return None
    label_str = str(label).strip().lower()
    mapping = {
        "not useful": 0,
        "useful": 1
    }
    if label_str in mapping:
        return mapping[label_str]
    try:
        return int(label)  # fallback if numeric
    except Exception:
        return None

# ------------------------------
# Graph builder (toy example)
# ------------------------------
def build_graph(code, comment, label):
    """
    Convert code + comment into a graph.
    Right now we just build a simple graph:
    - Nodes = characters of code
    - Comment stored in graph metadata
    """
    if not isinstance(code, str) or not isinstance(comment, str):
        return None

    # Example: char-level graph (you can swap with AST parser later)
    G = nx.path_graph(len(code))
    x = torch.ones((G.number_of_nodes(), 1))  # feature: all 1s

    # Make PyG Data object
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    if edge_index.numel() == 0:
        return None

    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
    )
    return data

# ------------------------------
# Main function
# ------------------------------
def main():
    input_csv = "data/raw/data.csv"
    output_dir = "data/graphs"
    os.makedirs(output_dir, exist_ok=True)

    print(f"📂 Loading dataset: {input_csv}")
    df = pd.read_csv(input_csv)

    # Rename columns consistently
    df = df.rename(columns={
        "Comments": "comment",
        "Surrounding Code Context": "code",
        "Class": "label"
    })

    print(f"📊 Total rows: {len(df)}")

    skipped = 0
    saved = 0

    for i, row in tqdm(df.iterrows(), total=len(df)):
        label = safe_label(row["label"])

        # ✅ FIXED: only skip when label is None
        if label is None:
            print(f"⚠️ Skipping row {i}: Invalid label: {row['label']}")
            skipped += 1
            continue

        graph = build_graph(row["code"], row["comment"], label)
        if graph is None:
            print(f"⚠️ Skipping row {i}: Could not build graph")
            skipped += 1
            continue

        # Save graph
        torch.save(graph, os.path.join(output_dir, f"graph_{i}.pt"))
        saved += 1

    print(f"\n✅ Finished! Saved {saved} graphs to {output_dir}")
    print(f"⚠️ Skipped: {skipped} rows out of {len(df)}")

if __name__ == "__main__":
    main()
