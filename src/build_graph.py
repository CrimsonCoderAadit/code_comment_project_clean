import os
import pandas as pd
import networkx as nx

def build_graphs(input_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(input_file)

    for i, row in df.iterrows():
        comment = row.get("Comments", "")
        code = row.get("Surrounding Code Context", "")
        label = row.get("Class", "Unknown")

        G = nx.DiGraph()

        # 🔑 Add consistent node attributes
        G.add_node(
            f"comment_{i}",
            type="comment",
            text=str(comment) if pd.notna(comment) else ""
        )
        G.add_node(
            f"code_{i}",
            type="code",
            text=str(code) if pd.notna(code) else ""
        )

        # Edge between comment and code
        G.add_edge(f"comment_{i}", f"code_{i}")

        # Save graph with label in filename
        label_str = "Useful" if str(label).lower() == "useful" else "NotUseful"
        out_path = os.path.join(out_dir, f"graph_{i}_label{label_str}.gpickle")
        nx.write_gpickle(G, out_path)

    print(f"✅ Processed {len(df)} graphs into {out_dir}")


def main():
    input_file = "data/Code_Comment_Seed_Data.csv"
    out_dir = "data/graphs"
    build_graphs(input_file, out_dir)


if __name__ == "__main__":
    main()
