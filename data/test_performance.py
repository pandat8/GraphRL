import torch
import torch.optim as optm
import torch.nn.functional as F
import time

import numpy as np
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
from data.graph import Graph
from utils import utils
from collections import namedtuple
import matplotlib.pyplot as plt

# Mont Carlo methods
class Test:

    def __init__(self,  test_dataset=None, weight_d=5e-4, max_grad_norm=2, use_cuda=False):

        self.test_dataset = test_dataset
        self.weight_d = weight_d
        self.max_grad_norm = max_grad_norm
        self.use_cuda = use_cuda

        self.test_loader = DataLoader(test_dataset, shuffle=True, batch_size=1, collate_fn=lambda x: x)

        self.epochs = 0
        self.beta = 0.9
        self.eps = np.finfo(np.float32).eps.item()


    def test_heuristics(self):
        """
        Evaluation function
        :param model: network model
        :param data_loader: dataset loader depending on validation or test
        :param features: initial feature vector of graph
        :param is_cuda:
        :param validation: True if validation(by default), False if test
        :return: averaged loss per graph
        """

        n_graphs_proceed = 0
        n_sampling = 5

        for X in self.test_loader:
            _ratio = 0

            for x in X:
                min_degree_all = []
                one_step_all = []
                random_all = []
                # graph_size = []

                for i in range(n_sampling):

                    n_e_mindegree = 0  # number of added edges
                    n_e_random = 0
                    n_e_onestep = 0

                    q_mindegree = np.zeros(x.n - 1, dtype=int)  # index of ordering
                    # q2_mindegree = np.zeros(x.n - 1, dtype=int)  # degree value of ordering
                    q3_mindegree = np.zeros(x.n - 1, dtype=int)  # number of edges added each step

                    q_random = np.zeros(x.n - 1, dtype=int)  # index of ordering
                    q3_random = np.zeros(x.n - 1, dtype=int)  # number of edges added each step

                    q_onestep = np.zeros(x.n - 1, dtype=int)  # index of ordering
                    q3_onestep = np.zeros(x.n - 1, dtype=int) # number of edges to make neighbour-clique of ordering

                    x1 = Graph(x.M)
                    x4 = Graph(x.M)

                    x2 = Graph(x.M)

                    for i in range(x.n - 2):

                        # choose the node with minimum degree
                        node_chosen, d_min = x1.min_degree(x1.M)

                        q_mindegree[i] = node_chosen
                        # q2_mindegree[i] = d_min

                        # choose the node with one step greedy
                        q_onestep[i] = x2.onestep_greedy()

                        # choose the node randomly
                        q_random[i] = np.random.randint(low=0, high=x4.n)


                        # eliminate the node chosen
                        q3_mindegree[i] = x1.eliminate_node(q_mindegree[i], reduce=True)
                        n_e_mindegree += q3_mindegree[i]

                        q3_onestep[i] = x2.eliminate_node(q_onestep[i], reduce=True)
                        n_e_onestep += q3_onestep[i]

                        q3_random[i] = x4.eliminate_node(q_random[i], reduce=True)
                        n_e_random += q3_random[i]

                        # n_e_baseline = n_e_mindegree

                        # print( 'graph {:04d}'.format(n_graphs_proceed),
                        #        'size {}'.format(x.n),
                        #        'min_degree {}'.format(n_e_mindegree),
                        #        'one_step {}'.format(n_e_onestep),
                        #        'random {}'.format(n_e_random),
                        #     )



                    min_degree_all.append(n_e_mindegree)
                    one_step_all.append(n_e_onestep)
                    random_all.append(n_e_random)
                    # graph_size.append(x.n)


                min_degree_all = np.array(min_degree_all).reshape(-1)
                one_step_all = np.array(one_step_all).reshape(-1)
                random_all = np.array(random_all).reshape(-1)
                # graph_size_all = np.array(graph_size).reshape(-1)

                min_degree_av = np.sum(min_degree_all) / n_sampling
                one_step_av = np.sum(one_step_all) / n_sampling
                random_av = np.sum(random_all) / n_sampling
                # graph_size_av = np.sum(graph_size_all) / n_graphs_proceed
                print(
                    'graph {:04d}'.format(n_graphs_proceed),
                    'size_av {}'.format(x.n),
                    'min_degree_av {}'.format(min_degree_av),
                    'one_step_av {}'.format(one_step_av),
                    'random_av {}'.format(random_av)
                )
            n_graphs_proceed += len(X)


        print('test finished')



