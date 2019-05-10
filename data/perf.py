from graph import Graph
import time

f = 0.7
ns = 1

for n in [10, 20, 50, 100, 200]:

    g = Graph.erdosrenyi(n, f)
    ts = time.time()
    for i in range(ns):
        g.onestep_greedy_d()
    te = time.time()

    print("os-greedy: %4d %8.3fs" % (n, (te-ts) / ns))
