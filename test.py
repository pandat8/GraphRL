import torch
import scipy.io as spio
import numpy as np
from torch.utils.data import Dataset
from data.graph import Graph

folder = 'data/UFSMcollection/c-'+str(20)+'.mtx'
adj_matrix = spio.mmread(folder)
adj_matrix = adj_matrix.todense()
np.fill_diagonal(adj_matrix, 1)
adj_matrix = adj_matrix.astype(np.uint8)
g = Graph(adj_matrix)
print(g.M.diagonal())
