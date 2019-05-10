# This code must be compiled before being imported into Python
# To do so, run the following command in this directory:
#   python setup.py build_ext --inplace

import numpy as np
cimport numpy as np

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t


def c_onestep_greedy_d(np.ndarray[DTYPE_t, ndim=2] A):
    """
    Return a uniform distribution over nodes whose elimination adds fewer edges.

    # Arguments
    - `A`: Array of uint8
        The adjacency matrix of the graph

    # Return
    - `p`: Array of Float
        Uniform probability distribution over the nodes to be eliminated.
    """
    cdef int m, n
    m = A.shape[0]
    n = A.shape[1]

    assert m == n  # check that input matrix is square
    assert A.dtype == DTYPE

    s = np.zeros(n)

    cdef int e, i, j, k, smin

    for i in range(n):
        e = 0

        for j in range(n):

            if A[i, j] == 0:
                continue

            for k in range(j+1, n):  # only consider j < k

                if A[i, k] == 0:
                    continue
                
                s[i] += A[j,k]

    s_min = np.min(s)
    p = (s == s_min)
    p = (p / np.sum(p))

    return p