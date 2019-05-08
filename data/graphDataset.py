import torch
from torch.utils.data import Dataset
from data.graph import Graph
from utils.utils import erdosrenyi

class GraphDataset(Dataset):

    def __init__(self, n_nodes, n_graphs, random_seed = 32):
        """
        Generate a graph dataset with specific graph instances
        :param n_graphs: number of graphs
        :param n_nodes:  number of nodes
        :param random_seed:
        """
        super(GraphDataset, self).__init__()
        torch.manual_seed(random_seed)

        self.dataset = []

        for l in range(n_graphs): # feed random Erdos-Renyi graph instances into dataset
            g = Graph.erdosrenyi(n_nodes)
            self.dataset.append(g)

        self.size = len(self.dataset)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.dataset[idx]

class AdjacencyDataset(Dataset):

    def __init__(self, n_nodes, n_graphs, random_seed = 32):
        """
        Generate a graph dataset with specific graph instances
        :param n_graphs: number of graphs
        :param n_nodes:  number of nodes
        :param random_seed:
        """
        super(AdjacencyDataset, self).__init__()
        torch.manual_seed(random_seed)

        self.dataset = []

        for l in range(n_graphs): # feed random Erdos-Renyi graph instances into dataset
            M = erdosrenyi(n_nodes)
            self.dataset.append(M)

        self.size = len(self.dataset)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.dataset[idx]
