import os
import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset

class CodeCommentGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # not used, we directly read .gpickle

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        data_list = []
        graph_dir = os.path.join(self.root, "graphs")

        for fname in os.listdir(graph_dir):
            if fname.endswith(".gpickle"):
                G = nx.read_gpickle(os.path.join(graph_dir, fname))
                
                # Convert to PyG Data object
                data = from_networkx(G)

                # Graph label from filename (Useful or NotUseful)
                if "Useful" in fname:
                    data.y = torch.tensor([1], dtype=torch.long)
                else:
                    data.y = torch.tensor([0], dtype=torch.long)

                data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
