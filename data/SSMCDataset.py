import torch
import scipy.io as spio
import numpy as np
from torch.utils.data import Dataset
from data.graph import Graph

class SSMCDataset(Dataset):

    def __init__(self, random_seed = 33):
        """
        Generate a graph dataset with specific graph instances
        :param n_graphs: number of graphs
        :param n_nodes:  number of nodes
        :param random_seed:
        """
        super(SSMCDataset, self).__init__()
        torch.manual_seed(random_seed)

        self.dataset = []

        fxm3_6 = spio.mmread('data/HBcollection/fxm3_6.mtx')
        fxm3_6 = fxm3_6.todense()
        np.fill_diagonal(fxm3_6, 0)
        fxm3_6 = fxm3_6.astype(np.uint8)
        g = Graph(fxm3_6)
        self.dataset.append(g)

        self.size = len(self.dataset)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.dataset[idx]
