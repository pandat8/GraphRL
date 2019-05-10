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

SavedAction = namedtuple('SavedAction', ['log_prob', 'value_current'])

# Mont Carlo methods
class Train_SupervisedLearning:

    def __init__(self, model, heuristic = 'min_degree', train_dataset=None, val_dataset=None, test_dataset=None, weight_d=5e-4, max_grad_norm=2, use_cuda=False):
        self.model = model
        self.heuristic = heuristic
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.weight_d = weight_d
        self.max_grad_norm = max_grad_norm
        self.use_cuda = use_cuda

        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
        self.val_loader = DataLoader(val_dataset, shuffle=True, batch_size=1, collate_fn=lambda x: x)
        self.test_loader = DataLoader(test_dataset, shuffle=True, batch_size=1, collate_fn=lambda x: x)

        self.epochs = 0
        self.beta = 0.9
        self.eps = np.finfo(np.float32).eps.item()

    def test(self):
        """
        Evaluation function
        :param model: network model
        :param data_loader: dataset loader depending on validation or test
        :param features: initial feature vector of graph
        :param is_cuda:
        :param validation: True if validation(by default), False if test
        :return: averaged loss per graph
        """
        self.model.eval()

        n_graphs_proceed = 0
        ratio = []
        ratio_g2r = []
        for X in self.test_loader:
            _ratio = 0
            for x in X:
                n_e_mindegree = 0  # number of added edges
                n_e_random = 0
                # n_e_onestep = 0

                q_mindegree = np.zeros(x.n - 1, dtype=int)  # index of ordering
                q2_mindegree = np.zeros(x.n - 1, dtype=int)  # degree value of ordering
                q3_mindegree = np.zeros(x.n - 1, dtype=int)  # number of edges added each step

                q_random = np.zeros(x.n - 1, dtype=int)  # index of ordering
                q3_random = np.zeros(x.n - 1, dtype=int)  # number of edges added each step

                # q_onestep = np.zeros(x.n - 1, dtype=int)  # index of ordering
                # q2_onestep = np.zeros(x.n - 1, dtype=int) # number of edges to make neighbour-clique of ordering

                x1 = Graph(x.M)
                x4 = Graph(x.M)

                # x2 = Graph(x.M)

                for i in range(x.n - 2):
                    # choose the node with minimum degree
                    if self.heuristic == 'min_degree':
                        node_chosen, d_min = x1.min_degree(x1.M)
                    elif self.heuristic == 'one_step_greedy':
                        node_chosen, d_min = x1.onestep_greedy()
                    q_mindegree[i] = node_chosen
                    q2_mindegree[i] = d_min

                    # random eliminate a node
                    q_random[i] = np.random.randint(low=0, high=x4.n)
                    q3_random[i] = x4.eliminate_node(q_random[i], reduce=True)
                    n_e_random += q3_random[i]

                    # # choose the node with one step greedy
                    # q_onestep[i], q2_onestep = x2.onestep_greedy()
                    # eliminate the node chosen
                    q3_mindegree[i] = x1.eliminate_node(q_mindegree[i], reduce=True)
                    n_e_mindegree += q3_mindegree[i]

                n_e_baseline = n_e_mindegree
                print( '{} '.format(self.heuristic),
                  'inserted edges{}'.format(n_e_baseline))
                # samles multi solutions by GCN and pick the lowest cost one
                samle_gcn = 1

                edges_total_samples = np.zeros(samle_gcn)
                q_gcn_samples = np.zeros([samle_gcn, x.n - 1], dtype=int)  # index of ordering given by GCN
                edges_added = np.zeros([samle_gcn, x.n - 1], dtype=int)  # number of edges added each step
                for j in range(samle_gcn):
                    x3 = Graph(x.M)
                    for i in range(x.n - 2):

                        # choose the node with GCN
                        features = np.ones([x3.n, 1], dtype=np.float32)
                        M_gcn = torch.FloatTensor(x3.M)
                        M_gcn = utils.to_sparse(M_gcn)  # convert to coo sparse tensor
                        features = torch.FloatTensor(features)
                        if self.use_cuda:
                            M_gcn = M_gcn.cuda()
                            features = features.cuda()
                        output = self.model(features, M_gcn)
                        output = output.view(-1)
                        # q_gcn_samples[j, i] = np.argmax(
                        #     np.array(output.detach().cpu().numpy()))  # choose the node given by GCN

                        # output = np.array(output.detach().cpu().numpy())
                        # output = np.exp(output)
                        # q_gcn_samples[j, i] = np.random.choice(a=x3.n, p=output)

                        # output = torch.log(output)
                        # output = torch.exp(logits=output)
                        m = Categorical(logits=output)
                        q_gcn_samples[j, i] = m.sample()
                        # q_gcn_samples[j, i] = np.random.choice(a=x3.n, p=np.array(output.detach().cpu().numpy()))

                        edges_added[j, i] = x3.eliminate_node(q_gcn_samples[j, i],
                                                              reduce=True)  # eliminate the node and return the number of edges added
                        edges_total_samples[j] += edges_added[j, i]
                k = np.argmin(edges_total_samples)
                n_e_gcn = edges_total_samples[k]

                q_gcn = q_gcn_samples[k, :]
                q3_gcn = edges_added[k, :]

                _ratio = n_e_gcn / n_e_baseline
                _ratio_g2r = n_e_gcn / n_e_random
                ratio.append(_ratio)
                ratio_g2r.append(_ratio_g2r)
                # print('GCN number of edges {}'.format(np.sum(q3_gcn)))
                # print('Ran number of edges {}'.format(n_e_random))

                # n_e_onestep += x2.eliminate_node(q_onestep[i], reduce=True)

                # num_e = torch.IntTensor(x.num_e)

                # print('epoch {:04d}'.format(epoch),
                # 'min_degree number of edges {}'.format(n_e_mindegree))
                # print('epoch {:04d}'.format(epoch),
                # 'mindegree elimination ordering {}'.format(q_mindegree))
                # print('epoch {:04d}'.format(epoch),
                # 'mindegree elimination edge add {}'.format(q3_mindegree))
                #
                # print('epoch {:04d}'.format(epoch),
                # 'GCN number of edges {}'.format(n_e_gcn))
                # print('epoch {:04d}'.format(epoch),
                # 'GCN elimination ordering {}'.format(q_gcn))
                # print('epoch {:04d}'.format(epoch),
                # 'GCN elimination edge add {}'.format(q3_gcn))

                # print('epoch {:04d}'.format(epoch),
                # 'one_step_greedy number of edges {}'.format(n_e_onestep))
                # print('epoch {:04d}'.format(epoch),
                # 'one_step_greedy elimination ordering {}'.format(q_onestep))
                # print('epoch {:04d}'.format(epoch),
                # 'one_step_greedy {}'.format(q2_onestep))
            n_graphs_proceed += len(X)
        ratio = np.array(ratio).reshape(-1)
        ratio_g2r = np.array(ratio_g2r).reshape(-1)

        total_ratio = np.sum(ratio)
        total_ratio_g2r = np.sum(ratio_g2r)

        min_ratio = np.min(ratio)
        max_ratio = np.max(ratio)
        av_ratio = total_ratio / n_graphs_proceed

        min_ratio_g2r = np.min(ratio_g2r)
        max_ratio_g2r = np.max(ratio_g2r)
        av_ratio_g2r = total_ratio_g2r / n_graphs_proceed

        return ratio, av_ratio, max_ratio, min_ratio, ratio_g2r, av_ratio_g2r, max_ratio_g2r, min_ratio_g2r

    def train(self, lr =0.001, epochs=1):
        """
        Training function
        :param model: neural network model
        :param opt: optimizaiton function
        :param train_loader: training dataset loader
        :param features: features vector
        :param is_cuda:
        :return: averaged loss per graph
        """

        opt = optm.Adam(self.model.parameters(), weight_decay=self.weight_d, lr=lr)
        self.model.train()

        if self.use_cuda:
            plt.switch_backend('agg')

        total_acc_train = 0
        n_graphs_proceed = 0
        # n_iteration = 0
        total_loss_train = []
        t0 = time.clock()
        t1 = 0
        t2 = 0
        t3 = 0
        t4 = 0
        t5 = 0

        t = []

        ave_gcn = []
        min_gcn = []
        max_gcn = []

        ave_mind = []
        min_mind = []
        max_mind = []

        ave_ratio_gcn2mind = []
        min_ratio_gcn2mind = []
        max_ratio_gcn2mind = []

        for epoch in range(epochs):
            gcn_greedy = []
            mind = []
            ratio_gcn2mind = []

            av_loss_train = 0 # loss per epochs
            for X in self.train_loader:
                for x in X:

                    n = x.n
                    x1 = Graph(x.M)
                    total_loss_train_1graph = 0
                    depth = np.min([n - 2, 300])
                    # edges_total = 0
                    i = 0
                    while (i < depth) and (x1.n > 2):

                        node_selected, d_min = x1.min_degree(x1.M)
                        if not (d_min == 1 or d_min == 0):
                            i += 1
                            features = np.ones([x1.n, 1], dtype=np.float32)
                            m = torch.FloatTensor(x1.M)
                            _t1 = time.clock()
                            m = utils.to_sparse(m)  # convert to coo sparse tensor
                            t1 += time.clock() - _t1
                            features = torch.FloatTensor(features)

                            _t3 = time.clock()
                            if self.heuristic == 'min_degree':
                                distribution_labels = x1.min_degree_d()
                            elif self.heuristic == 'one_step_greedy':
                                distribution_labels = x1.onestep_greedy_d()

                            # distribution_labels = np.log(distribution_labels)

                            t3 += time.clock() - _t3
                            distribution_labels = torch.FloatTensor(distribution_labels)

                            # node_chosen, z = x1.min_degree(x1.M)  # get the node with minimum degree as label
                            # node_chosen = torch.from_numpy(np.array(node_chosen))  # one-hot coding
                            # node_chosen = node_chosen.reshape(1)
                            _t4 = time.clock()
                            if self.use_cuda:
                                m = m.cuda()
                                features = features.cuda()
                                distribution_labels = distribution_labels.cuda()

                            t4 += time.clock() - _t4

                            output = self.model(features, m)
                            output = output.view(-1)


                            # m = Categorical(output)
                            # node_selected = m.sample()
                            # node_selected = torch.LongTensor([[node_selected]])
                            # m.probs.zero_()
                            # m.probs.scatter_(1, node_selected, 1)

                            loss_train = F.kl_div(output, distribution_labels) # get the negetive likelyhood
                            total_loss_train_1graph += loss_train.item()
                            _t5 = time.clock()
                            opt.zero_grad()
                            loss_train.backward()
                            opt.step()
                            t5 += time.clock() - _t5


                            # action_gcn = np.argmax(np.array(output.detach().cpu().numpy()))  # choose the node given by GCN
                            # output = np.array(output.detach().cpu().numpy())
                            # output = np.exp(output)
                            # action_gcn = np.random.choice(a=x1.n, p=output)

                            # output = torch.log(output)
                            # output = torch.exp(output)
                            m = Categorical(logits=output) # logits=probs
                            action_gcn = m.sample()

                            _t2 = time.clock()
                            edges_added = x1.eliminate_node(action_gcn, reduce=True)
                            # edges_total +=edges_added
                            t2 += time.clock() - _t2
                        else:
                            reward = x1.eliminate_node(node_selected, reduce=True)

                av_loss_train += total_loss_train_1graph
            print('epochs {}'.format(epoch), 'loss {}'.format(av_loss_train))

            for X in self.val_loader:
                for x in X:

                    n = x.n
                    self.model.eval()
                    # ratio_gcn2mind = []
                    # ratio_gcn2rand = []

                    rewards_mindegree = 0  # number of added edges
                    # rewards_random = 0
                    x_mind = Graph(x.M)
                    # x_rand = Graph(x.M)
                    x_model = Graph(x.M)

                    # loop for training while eliminating a graph iteratively
                    i = 1
                    depth = np.min([n - 2, 300])
                    rewards_gcn_greedy = np.zeros(1)
                    while (i < depth) and (x_model.n > 2):

                        # baseline1: compute return of min degree
                        # if i % 100 == 0:
                        #     print('iterations {}'.format(i))
                        if self.heuristic == 'min_degree':
                            node_chosen, d_min = x_mind.min_degree(x_mind.M)
                        elif self.heuristic == 'one_step_greedy':
                            node_chosen, d_min = x_mind.onestep_greedy()
                        # node_mind, d_min = x_mind.min_degree(x_mind.M)
                        rewards_mindegree += x_mind.eliminate_node(node_chosen, reduce=True)

                        # baseline2: compute return of random
                        # rewards_random += x_rand.eliminate_node(np.random.randint(low=0, high=x_rand.n), reduce=True)

                        # call actor-critic model

                        node_selected, d_min = x_model.min_degree(x_model.M)
                        if not (d_min == 1 or d_min == 0):
                            i += 1
                            features = np.ones([x_model.n, 1], dtype=np.float32)
                            M_gcn = torch.FloatTensor(x_model.M)
                            features = torch.FloatTensor(features)
                            M_gcn = utils.to_sparse(M_gcn)  # convert to coo sparse tensor

                            if self.use_cuda:
                                M_gcn = M_gcn.cuda()
                                features = features.cuda()

                            probs = self.model(features, M_gcn)
                            probs = probs.view(-1)
                            # probs = torch.exp(probs)
                            m = Categorical(logits=probs) # logits=probs
                            q_gcn_samples = m.sample()
                            edges_added = x_model.eliminate_node(q_gcn_samples,
                                                                reduce=True)  # eliminate the node and return the number of edges added
                            rewards_gcn_greedy += edges_added
                        else:
                            reward = x_model.eliminate_node(node_selected, reduce=True)

                    # print('graph {:04d}'.format(n_graphs_proceed), 'epoch {:04d}'.format(epoch),
                    #       'gcn2mind ratio {}'.format(_ratio_gcn2mind),
                    #       'value {}'.format(saved_actions[0].value_current),
                    #       'R {}'.format(returns[0]))
                    # print('graph {:04d}'.format(n_graphs_proceed), 'epoch {:04d}'.format(epoch),
                    #       'gcn2rand ratio {}'.format(_ratio_gcn2rand))

                    _ratio_gcn2mind = rewards_gcn_greedy / rewards_mindegree
                    # _ratio_gcn2rand = rewards_gcn / rewards_random
                    gcn_greedy.append(rewards_gcn_greedy)
                    mind.append(rewards_mindegree)
                    # rand.append(rewards_random)
                    ratio_gcn2mind.append(_ratio_gcn2mind)
                    # ratio_gcn2rand.append(_ratio_gcn2rand)

                # n_graphs_proceed += len(X)

            gcn_greedy = np.array(gcn_greedy).reshape(-1)
            mind = np.array(mind).reshape(-1)
            # rand = np.array(rand).reshape(-1)
            ratio_gcn2mind = np.array(ratio_gcn2mind).reshape(-1)
            # ratio_gcn2rand = np.array(ratio_gcn2rand).reshape(-1)

            _ave_gcn = np.sum(gcn_greedy) / len(gcn_greedy)
            _min_gcn = np.min(gcn_greedy)
            _max_gcn = np.max(gcn_greedy)

            _ave_mind = np.sum(mind) / len(mind)
            _min_mind = np.max(mind)
            _max_mind = np.min(mind)

            # _ave_rand = np.sum(rand) / len(rand)
            # _min_rand = np.max(rand)
            # _max_rand = np.min(rand)

            _min_ratio_gcn2mind = np.min(ratio_gcn2mind)
            _max_ratio_gcn2mind = np.max(ratio_gcn2mind)
            _ave_ratio_gcn2mind = np.sum(ratio_gcn2mind) / len(ratio_gcn2mind)

            # _min_ratio_gcn2rand = np.min(ratio_gcn2rand)
            # _max_ratio_gcn2rand = np.max(ratio_gcn2rand)
            # _ave_ratio_gcn2rand = np.sum(ratio_gcn2rand) / len(ratio_gcn2rand)

            t.append(epoch)
            ave_gcn.append(_ave_gcn)
            min_gcn.append(_min_gcn)
            max_gcn.append(_max_gcn)
            ave_mind.append(_ave_mind)
            min_mind.append(_min_mind)
            max_mind.append(_max_mind)
            # ave_rand.append(_ave_rand)
            # min_rand.append(_min_rand)
            # max_rand.append(_max_rand)

            ave_ratio_gcn2mind.append(_ave_ratio_gcn2mind)
            min_ratio_gcn2mind.append(_min_ratio_gcn2mind)
            max_ratio_gcn2mind.append(_max_ratio_gcn2mind)

            # print('epochs {}'.format(epoch),'loss {}'.format(av_loss_train) )
            total_loss_train.append(av_loss_train)

            t_plot = np.array(t).reshape(-1)

            ave_gcn_plot = np.array(ave_gcn).reshape(-1)
            ave_mind_plot = np.array(ave_mind).reshape(-1)

            if self.use_cuda:
                plt.clf()
                plt.plot(t_plot, ave_gcn_plot, t_plot, ave_mind_plot)
                plt.legend(('GNN', self.heuristic),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
                           loc='upper right')  # 'GNN-initial', 'GNN-RL', 'min-degree'
                plt.title('Supervised learning curve with pretrain '+self.train_dataset.__class__.__name__+' (average number of filled edges)')
                plt.ylabel('number of fill-in')
                # plt.draw()
                plt.savefig(
                    './results/supervised'+str(lr)+'_'+self.heuristic+'_curve_g2m_number_gcn_logsoftmax_'+self.train_dataset.__class__.__name__+'_cuda.png')
                plt.clf()
            else:
                plt.clf()
                plt.plot(t_plot, ave_gcn_plot, t_plot, ave_mind_plot)
                plt.legend(('GNN-RL', 'GNN-RL-epsilon', 'min-degree'),
                           loc='upper right')
                plt.title('RL-MonteCarlo learning curve with pretrain ERG100 (average number of filled edges)')
                plt.ylabel('number of fill-in')
                # plt.draw()
                plt.savefig('./results/acmc001_learning_curve_g2m_number_gcn_non_pretrainERG100_with_epsilon05.png')
                plt.clf()

            print('epoch {:04d}'.format(epoch), 'gcn2'+self.heuristic,
                  'min_ratio {}'.format(_min_ratio_gcn2mind),
                  'max_ratio {}'.format(_max_ratio_gcn2mind),
                  'av_ratio {}'.format(_ave_ratio_gcn2mind))
            for name, param in self.model.named_parameters():
                print('parameter name {}'.format(name),
                    'parameter value {}'.format(param.data))


        gcn_greedy = np.array(ave_gcn).reshape(-1)
        ave_ratio_gcn2mind = np.array(ave_ratio_gcn2mind).reshape(-1)
        # ave_ratio_gcn2rand = np.array(ave_ratio_gcn2rand).reshape(-1)

        t = np.arange(0, epochs, 1)
        if self.use_cuda:
            plt.clf()
            plt.plot(t, ave_ratio_gcn2mind)
            plt.legend(('GNN-RL/'+self.heuristic),
                       loc='upper right')
            plt.title('Supervised learning curve ratio with pretrain '+self.train_dataset.__class__.__name__)
            plt.ylabel('fill-in ratio: gnn model/heuristic')
            plt.savefig(
                './results/supervised'+str(lr)+'_'+self.heuristic+'_curve_g2m_ratio_gcn_logsoftmax_'+self.train_dataset.__class__.__name__+'_cuda.png')
            plt.clf()
        else:
            plt.clf()
            plt.plot(t, ave_ratio_gcn2mind)
            plt.legend(('GNN-RL/mindegree'),
                       loc='upper right')
            plt.title('RL-MonteCarlo learning curve ratio with pretrain ERG100')
            plt.ylabel('fill-in ratio: gnn model/heuristic')
            plt.savefig('./results/acmc001_learning_curve_g2m_ratio_gcn_non_pretrainERG100_with_epsilon05.png')
            plt.clf()

        total_loss_train = np.array(total_loss_train).reshape(-1)

        t0 = time.clock() - t0

        return av_loss_train, t5, t4, t3, t2, t1, t0

