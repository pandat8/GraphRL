import torch
import numpy as np

def to_sparse(dense):
    x = torch.nonzero(dense)
    if list(x):
        indices = x.t()
        values = dense[indices[0], indices[1]] #
        sparse = torch.sparse.FloatTensor(indices, values, dense.size())
    else:
        sparse = torch.sparse.FloatTensor(dense.size())
    return sparse
