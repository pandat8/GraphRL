import itertools
import torch
import numpy as np
import pickle as pkl
import time

def save_dataset(filename, train, val, test):
    with open(filename, "wb") as f:
        pkl.dump([train,val, test], f)

def open_dataset(filename):
    with open(filename, "rb") as f:
        train, val, test  = pkl.load(f)
    return train, val, test

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

def elimination_step( M, k, reduce=True):
    """
    Delete node k from graph.

    # Arguments
    - `k`: Integer
        The index of the node to be deleted.

    # Returns
    - `M`: Dense Adj Matrix
        Graph after elimination.
    - `m`: Integer
        The number of edges that were addded to the graph.
    """

    n = M.shape[0]  # Number of nodes
    assert (0 <= k < n)  # node must exist in graph
    m = 0

    # Get neighbours of node i
    N = np.where(M[k, :])[0]

    # Add edges to make neighbouring nodes into a clique
    for (i, j) in itertools.combinations(N, 2):
        if M[i, j]:
            # edge already exists
            continue
        else:
            # create new edge ! Change to self.M_ex for one-step solution ordering
            M[i, j] = 1
            M[j, i] = 1
            m += 1
    zeros_row = np.zeros(n)
    # Remove node from the graph
    if reduce:
        M = np.delete(M, k, 0)
        M = np.delete(M, k, 1)
    else:
        zeros_row = np.zeros(n)
        M[k, :] = zeros_row
        M[:, k] = np.transpose(zeros_row)

    return M, m


def heuristic_min_degree(M):
    """
    Identify nodes with min degree and choose one of them by random
    :return: return the index of the index of the node chosen
    """
    d = np.sum(M, 1) # degree vector
    # d2 = d[np.where(d)] # remove zero elements corresponding to eliminated nodes
    d_min = np.min(d)  # minimum degree
    indices = np.array(np.nonzero(d == d_min))  # identify nodes with minimum degree
    node_chosen = np.random.choice(indices.reshape(-1))
    return node_chosen,d_min

def heuristic_onestep_greedy(M):
    """
    Identify node(s) whose elimination adds fewer edges.

    # Arguments

    # Returns
    - `p`: Array of Float
        Uniform probability distribution over the nodes to be eliminated.
    """
    n = M.shape[0]  # Number of nodes
    s = np.zeros(n, dtype=int)
    r = np.arange(n)

    for i in range(n):
        e = 0  # number of edges to add
        neighbours = r[M[:,i]==1]  # neighbours of node i
        for (j, k) in itertools.combinations(neighbours, 2):
            e += (1-M[j, k])
        s[i] = e

    s_min = np.min(s)
    indices = np.array(np.nonzero(s == s_min))  # identify nodes with minimum degree
    node_chosen = np.random.choice(indices.reshape(-1))

    return node_chosen, s_min
