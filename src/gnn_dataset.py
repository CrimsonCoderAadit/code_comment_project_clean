"""
Custom PyTorch Geometric dataset for loading graphs built by build_graph.py.
Ensures all graphs have consistent node features and labels.
"""

import os
import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

GRAPH_DIR = "data/graphs"


class CodeCommentGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.root = root
        super().__init__(root, transform, pre_transform)
        # Load processed dataset
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # No raw files needed (we already built graphs separately)
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        # No downloading required
        pass

    def process(self):
        print(f"Processing graphs from: {GRAPH_DIR}")
        files = [f for f in os.listdir(GRAPH_DIR) if f.endswith(".pt")]
        print(f"Found {len(files)} graphs to process...")

        data_list = []
        for f in tqdm(files, desc="Loading graphs"):
            try:
                data = torch.load(os.path.join(GRAPH_DIR, f), weights_only=False)

                # Ensure x exists and is numeric
                if not hasattr(data, "x") or data.x is None:
                    continue
                data.x = torch.as_tensor(data.x, dtype=torch.float)

                # Ensure y exists
                if not hasattr(data, "y") or data.y is None:
                    continue
                data.y = torch.as_tensor(data.y, dtype=torch.long)

                data_list.append(data)
            except Exception as e:
                print(f"⚠️ Skipping {f}: {e}")

        print(f"✅ Processed {len(data_list)} valid graphs.")
        if len(data_list) == 0:
            raise RuntimeError("No valid graphs found! Check build_graph.py output.")

        torch.save(self.collate(data_list), self.processed_paths[0])
