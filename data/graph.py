import itertools
import numpy as np
import random

from c_graph import \
    c_onestep_greedy_d


class Graph:
    """
    Undirected graphs.

    # Attributes
    - `n`: Integer
        The number of nodes in the graph.
    - `M`: Numpy boolean matrix
        The adjacency matrix of the graph.
    """
    def __init__(self, M):
        assert M.shape[0] == M.shape[1]  # M is a square matrix
        assert M.dtype == np.uint8  # M must be boolean

        self.n = M.shape[0]  # Number of nodes
        self.M = np.array(M).astype(np.uint8)  # Adjacency matrix
        self.M_ex = np.copy(self.M)  # Adjacency matrix of chordal extention
        self.num_e = 0
        self._degree = np.count_nonzero(self.M, axis=1)  # Degree of nodes

        d = np.sum(self.M, 1)
        d_min = np.min(d)  # minimum degree
        indices = np.nonzero(d == d_min)  # identify nodes with minimum degree
        self._min_degree = random.choice(indices)

    @classmethod
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
            for j in range(i+1, n):
                if np.random.rand() < p:
                    # create an edge between nodes i and j
                    M[i, j] = 1
                    M[j, i] = 1

        return cls(M)

    @classmethod
    def empty(cls, n):
        """
        Generate an empty graph with n nodes.

        # Arguments
        - `n`: Integer
            The number of nodes in the graph.
        """
        assert n >= 0  # Positive number of nodes

        M = np.zeros((n, n), dtype=bool)

        return(cls(M))

    def eliminate_node(self, k, reduce=True):
        """
        Delete node k from graph.

        # Arguments
        - `k`: Integer
            The index of the node to be deleted.

        # Returns
        - `m`: Integer
            The number of edges that were addded to the graph.
        """
        assert (0 <= k < self.n)  # node must exist in graph
        m = 0

        # Get neighbours of node i
        N = np.where(self.M[k, :])[0]

        # Add edges to make neighbouring nodes into a clique
        for (i, j) in itertools.combinations(N, 2):
            if self.M[i, j]:
                # edge already exists
                continue
            else:
                # create new edge!
                # Change to self.M_ex for one-step solution ordering
                self.M[i, j] = 1
                self.M[j, i] = 1
                m += 1
        zeros_row = np.zeros(self.n)
        # Remove node from the graph
        if reduce:
            self.n -= 1
            self.M = np.delete(self.M, k, 0)
            self.M = np.delete(self.M, k, 1)
        else:
            zeros_row = np.zeros(self.n)
            self.M[k, :] = zeros_row
            self.M[:, k] = np.transpose(zeros_row)

        return m

    def chordal_extension(self, q):
        """
        Build a chordal extension given the elimination ordering `q`.

        # Arguments
        - `q`: Array of integer
            The elimination order for building the chordal extension.

        # Returns
        - `C`: Graph
            A chordal extension of the graph.
        - `m`: Integer
            The number of edges that were added to the graph when building
                the chordal extension.
        """
        C = Graph(self.M)
        num_e = 0  # Number of additional edges

        # Eliminate nodes
        for k in q[:]:  # last node need not be eliminated
            # A = self.M
            num_e += C.eliminate_node(k, reduce=False)
        self.M_ex = C.M_ex
        self.num_e = num_e

    def elimination_ordering(self, heuristic='min_degree'):
        """
        Generate an elimination ordering according to.

        :return:
        """
        M = np.copy(self.M)
        q = np.zeros(self.n-1, dtype=int)
        q2 = np.zeros(self.n-1, dtype=int)
        zeros_row = np.zeros(self.n)
        for i in range(self.n-1):
            if heuristic == 'min_degree':
                q[i], q2[i] = self.min_degree(M)
            if heuristic == 'onestep_greedy':
                q[i], q2[i] = self.onestep_greedy() # Here wrong! you should send a copy of self.M to this call
                                                    # and keep self.M from changing
            M[q[i],:] = zeros_row
            M[:,q[i]] = np.transpose(zeros_row)
        return q, q2

    def min_degree(self, M):
        """
        Identify nodes with min degree and choose one of them by random.
        :return: return the index of the index of the node chosen
        """
        d = np.sum(M, 1)  # degree vector
        # d2 = d[np.where(d)] # remove zero elements corresponding to eliminated nodes
        d_min = np.min(d)  # minimum degree
        indices = np.array(np.nonzero(d == d_min))  # identify nodes with minimum degree
        node_chosen = np.random.choice(indices.reshape(-1))
        return node_chosen, d_min

    def onestep_greedy(self):
        """
        Identify node(s) whose elimination adds fewer edges.

        # Arguments

        # Returns
        - `node`: int
            Node to eliminate.
        """
        p = self.onestep_greedy_d()
        node = np.random.choice(self.n, 1, p)

        return node[0]

# the following part is for testing the distribution of min degree

    def min_degree_d(self):
        """
        Compute the distribution over min digree nodes

        # Arguments
        # Returns
        - `p`: Array of Float
            Uniform probability distribution over the nodes of minimum degree.
        """
        # Compute degrees
        d = np.sum(self.M, 1)  # d[i] is the degree of node i
        d_min = np.min(d)  # minimum degree

        p = (d == d_min)  # identify nodes with minimum degree
        p = (p / np.sum(p))  # normalize to get probability distribution

        return p

    @property
    def degree_d(self):
        """
        Calculate the degree of nodes and the distribution of degree.

        # return:
        A numpy array with degree of nodes
        """
        _degree = np.count_nonzero(self.M, axis=1)
        _degree_d = _degree/_degree.sum()
        return _degree_d

    @degree_d.setter
    def degree_d(self, value):
        _degree_d = value

    def onestep_greedy_d(self):
        """
        Identify node(s) whose elimination adds fewer edges.

        # Arguments

        # Returns
        - `p`: Array of Float
            Uniform probability distribution over the nodes to be eliminated.
        """
        return c_onestep_greedy_d(self.M)

    def onestep_d(self):
        """
        Identify node(s) whose elimination adds fewer edges.

        # Arguments

        # Returns
        - `p`: Array of Float
            Uniform probability distribution over the nodes.
        """
        s = np.zeros(self.n, dtype=int)
        r = np.arange(self.n)

        for i in range(self.n):
            e = 0  # number of edges to add
            neighbours = r[self.M[:, i] == 1]  # neighbours of node i
            for (j, k) in itertools.combinations(neighbours, 2):
                e += (1-self.M[j, k])
            s[i] = e

        # s_min = np.min(s)
        # p = (s == s_min)  # identify nodes with minimum score
        s = (s / np.sum(s))  # normalize to get probability distribution

        return s
