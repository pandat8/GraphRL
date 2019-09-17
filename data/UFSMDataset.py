import torch
import scipy.io as spio
import numpy as np
from torch.utils.data import Dataset
from data.graph import Graph
from utils.utils import is_symmetric
import os

class UFSMDataset(Dataset):

    def __init__(self, random_seed=34, input_dir=''):
        """
        Generate a graph dataset with specific graph instances
        :param n_graphs: number of graphs
        :param n_nodes:  number of nodes
        :param random_seed:
        """
        super(UFSMDataset, self).__init__()
        torch.manual_seed(random_seed)

        self.dataset = []

        for path, directories, files in os.walk(input_dir):
            for f in files:
                folder = os.path.join(path, f)
                adj_matrix = spio.mmread(folder)
                adj_matrix.data[:] = 1
                # print(adj_matrix)
                adj_matrix = adj_matrix.todense()
                np.fill_diagonal(adj_matrix, 0)
                if not is_symmetric(adj_matrix):
                    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
                adj_matrix = adj_matrix.astype(np.uint8)
                g = Graph(adj_matrix)
                self.dataset.append(g)

        self.size = len(self.dataset)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.dataset[idx]
