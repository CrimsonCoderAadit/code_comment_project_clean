import os
import glob
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx
import networkx as nx

class CodeCommentGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # we already have gpickles

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        # Nothing to download
        pass

    def process(self):
        data_list = []
        graph_files = glob.glob(os.path.join(self.root, "graphs", "*.gpickle"))

        for gf in graph_files:
            G = nx.read_gpickle(gf)

            # 🔑 Ensure all nodes have "type" and "text"
            for n, d in G.nodes(data=True):
                if "type" not in d:
                    d["type"] = "unknown"
                if "text" not in d:
                    d["text"] = ""

            data = from_networkx(G)

            # Extract label from filename
            if "Useful" in gf:
                data.y = torch.tensor([1], dtype=torch.long)
            else:
                data.y = torch.tensor([0], dtype=torch.long)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
