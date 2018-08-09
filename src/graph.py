import itertools
import numpy as np

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
        assert M.dtype == bool  # M must be boolean
        
        self.n = M.shape[0]  # Number of nodes
        self.M = np.copy(M)  # Adjacency matrix
        

    @classmethod
    def erdosrenyi(cls, n, p):
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
        M = np.zeros((n, n), dtype=bool)
        if p==0:
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
                # create new edge
                self.M[i, j] = 1
                self.M[j, i] = 1
                m += 1
        
        # Remove node from the graph
        if reduce:
            self.n -= 1
            self.M = np.delete(self.M, k, 0)
            self.M = np.delete(self.M, k, 1)

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
        m = 0  # Number of additional edges

        # Eliminate nodes
        for k in q[:-1]:  # last node need not be eliminated
            m += C.eliminate_node(k, reduce=False)

        return C, m

    def min_degree(self):
        """
        Identify nodes with minimum degree.

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

        