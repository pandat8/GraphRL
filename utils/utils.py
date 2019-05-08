import torch
import numpy as np
import time

def to_sparse(dense):

    x = torch.nonzero(dense)
    a = len(x.size())
    if a:
        indices = x.t()
        values = dense[indices[0], indices[1]] #
        sparse = torch.sparse.FloatTensor(indices, values, dense.size())

    else:
        sparse = torch.sparse.FloatTensor(dense.size())
    return sparse

def erdosrenyi(cls, n, p=0.7):
    """
    Generate a random Erdos-Renyi graph.

    # Arguments
    - `n`: Integer
        The number of nodes in the graph.
    - `p`: Float
        The probability that an edge (i, j) be in the graph.

    # Returns
    - ``: Graph
        A randomly-generated Erdos-Renyi graph.
    """
    assert (0.0 <= p <= 1.0)  # p is a probability
    assert (0 <= n)  # Positive number of nodes

    # Create adjacency matrix
    M = np.zeros((n, n), dtype=np.uint8)
    if p == 0:
        return cls(M)  # early return

    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < p:
                # create an edge between nodes i and j
                M[i, j] = 1
                M[j, i] = 1

    return M
