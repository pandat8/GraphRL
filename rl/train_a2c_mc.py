import numpy as np
import torch
import torch.optim as optm
import torch.nn.functional as F
from torch.distributions import Categorical

from torch.utils.data import Dataset, DataLoader
from data.graph import Graph
from utils import utils
from collections import namedtuple
import matplotlib.pyplot as plt
import copy
import time

SavedAction = namedtuple('SavedAction', ['log_prob', 'value_current'])

# Mont Carlo methods
class TrainModel_MC:

    def __init__(self, model, train_dataset, val_dataset, weight_d=5e-4, max_grad_norm=2, use_cuda=False):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.weight_d = weight_d
        self.max_grad_norm = max_grad_norm
        self.use_cuda = use_cuda

        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
        self.val_loader = DataLoader(val_dataset, shuffle=True, batch_size=1, collate_fn=lambda x: x)

        self.epochs = 0
        self.beta = 0.9
        self.eps = np.finfo(np.float32).eps.item()


    def train_and_validate(self, n_epochs, lr_actor, lr_critic, use_critic, gamma=0.99):
        actor_initial = copy.deepcopy(self.model.actor)

        self.actor_optim = optm.Adam(self.model.actor.parameters(),  weight_decay=self.weight_d, lr=lr_actor)

        print('Use Critic:')
        print(use_critic)
        if use_critic:
            self.critic_optim = optm.Adam(self.model.critic.parameters(), weight_decay=self.weight_d, lr=lr_critic)
            self.critic_loss_criterion = torch.nn.MSELoss()
        else:
            baseline = torch.zeros(1)
            # if self.use_cuda:
            #     baseline = baseline.cuda()

        if self.use_cuda:
            plt.switch_backend('agg')

        ave_ratio_gcn2mind = []
        min_ratio_gcn2mind = []
        max_ratio_gcn2mind = []
        # ave_ratio_gcn2rand = []
        # min_ratio_gcn2rand = []
        # max_ratio_gcn2rand = []

        t = []
        ave_gcn_sto = []
        ave_gcn = []
        min_gcn = []
        max_gcn = []

        ave_mind = []
        min_mind = []
        max_mind = []

        # ave_rand = []
        # min_rand = []
        # max_rand = []

        for epoch in range(n_epochs):

            gcn_greedy = []
            gcn_sto = []

            mind = []
            # rand = []
            ratio_gcn2mind = []
            # ratio_gcn2rand = []
            n_graphs_proceed = 0
            # for batch_id, sample_batch in enumerate(self.train_loader):
            for X in self.train_loader:
                for x in X:

                    n = x.n
                    self.model.train()
                    # ratio_gcn2mind = []
                    # ratio_gcn2rand = []


                    rewards_mindegree = 0  # number of added edges
                    # rewards_random = 0
                    # x_mind = Graph(x.M)
                    # x_rand = Graph(x.M)
                    x_rl = Graph(x.M)
                    x_test = Graph(x.M)

                    # loop for training while eliminating a graph iteratively
                    i = 1
                    depth  = np.min([n-2, 300])
                    rewards_gcn_greedy = np.zeros(1)
                    while (i<depth) and (x_rl.n > 2):

                        # baseline1: compute return of min degree
                        # if i % 100 == 0:
                        #     print('iterations {}'.format(i))
                        # node_mind, d_min = x_mind.min_degree(x_mind.M)
                        # rewards_mindegree += x_mind.eliminate_node(node_mind, reduce=True)

                        # baseline2: compute return of random
                        # rewards_random += x_rand.eliminate_node(np.random.randint(low=0, high=x_rand.n), reduce=True)

                        # call actor-critic model

                        node_selected, d_min = x_rl.min_degree(x_rl.M)
                        if not (d_min==1 or d_min ==0 ):
                            i += 1
                            action, log_prob, reward, value_current, value_next, x_rl = self.model(x_rl) # forward propagation,action: node selected, reward: nb edges added
                            self.model.rewards.append(reward)
                            self.model.actions.append(action)
                            self.model.saved_actions.append(SavedAction(log_prob, value_current))
                        else:
                            reward = x_rl.eliminate_node(node_selected, reduce=True)
                            # reward = 2

                    R = torch.zeros(1)
                    # if self.use_cuda:
                    #     R = R.cuda()

                    actor_losses = []
                    critic_losses = []
                    returns = []

                    # compute sampled return for each step
                    for r in self.model.rewards[::-1]:
                        R = r + gamma * R
                        returns.insert(0, R)
                    returns = torch.tensor(returns)

                    returns = returns+1
                    returns = 1/returns

                    returns = (returns - self.model.epsilon*1420)/(1-self.model.epsilon)
                    # returns = returns / (returns.std() + self.eps)
                    returns = (returns - returns.mean()) / (returns.std() + self.eps)
                    saved_actions = self.model.saved_actions
                    # compute cummulated loss of actor and critic of one graph
                    gamma_t = 1
                    for (log_prob, value_current), R in zip(saved_actions, returns):
                        if use_critic:
                            advantage = R - value_current
                            critic_losses.append(-value_current* advantage)
                            # critic_losses.append(self.critic_loss_criterion(value_current, torch.Tensor([R.detach()])))
                        else:
                            advantage = R - baseline
                        if self.use_cuda:
                            advantage = advantage.cuda()
                        actor_losses.append(-gamma_t*log_prob * advantage.detach())  # the return here is discounted nb of added edges,
                                                                   # hence, it actually represents loss
                        gamma_t = gamma_t*gamma

                    # step update of actor
                    self.actor_optim.zero_grad()
                    actor_loss = torch.stack(actor_losses).sum()
                    actor_loss.backward(retain_graph=True)
                    self.actor_optim.step()
                    print('epochs {}'.format(epoch), 'loss {}'.format(actor_loss))

                    # step update of critic
                    if use_critic:
                        self.critic_optim.zero_grad()
                        critic_closs = torch.stack(critic_losses).sum()
                        critic_closs.backward()
                        self.critic_optim.step()
                    else:
                        baseline = baseline.detach()

                    # self.model.eval()
                    # i = 1
                    # while i < depth:
                    #     # gcn-greedy
                    #     node_selected, d_min = x_test.min_degree(x_test.M)
                    #     if not (d_min == 1 or d_min == 0):
                    #         i += 1
                    #
                    #         features = np.ones([x_test.n, 1], dtype=np.float32)
                    #         M_gcn = torch.FloatTensor(x_test.M)
                    #         features = torch.FloatTensor(features)
                    #         M_gcn = utils.to_sparse(M_gcn)  # convert to coo sparse tensor
                    #         if self.use_cuda:
                    #             M_gcn = M_gcn.cuda()
                    #             features = features.cuda()
                    #         probs = self.model.actor(features, M_gcn)
                    #         probs = probs.view(-1)
                    #
                    #
                    #         m = Categorical(probs)
                    #         q_gcn_samples = m.sample()
                    #         # q_gcn_samples = np.argmax(np.array(output.detach().cpu().numpy()))
                    #         edges_added = x_test.eliminate_node(q_gcn_samples,
                    #                                             reduce=True)  # eliminate the node and return the number of edges added
                    #         rewards_gcn_greedy += edges_added
                    #     else:
                    #         reward = x_test.eliminate_node(node_selected, reduce=True)


                    # rewards_gcn_sto = sum(self.model.rewards)
                    #
                    # _ratio_gcn2mind = rewards_gcn_sto / rewards_mindegree
                    # # _ratio_gcn2rand = rewards_gcn / rewards_random
                    # gcn_greedy.append(rewards_gcn_greedy)
                    # gcn_sto.append(rewards_gcn_sto)
                    # mind.append(rewards_mindegree)
                    # # rand.append(rewards_random)
                    # ratio_gcn2mind.append(_ratio_gcn2mind)
                    # # ratio_gcn2rand.append(_ratio_gcn2rand)

                    del self.model.rewards[:]
                    del self.model.actions[:]
                    del self.model.saved_actions[:]

                # n_graphs_proceed += len(X)

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
                    x_rl = Graph(x.M)
                    x_test = Graph(x.M)

                    # loop for training while eliminating a graph iteratively
                    i = 1
                    depth = np.min([n - 2, 300])
                    rewards_gcn_greedy = np.zeros(1)
                    while (i < depth) and (x_rl.n > 2):

                        # baseline1: compute return of min degree
                        # if i % 100 == 0:
                        #     print('iterations {}'.format(i))
                        node_mind, d_min = x_mind.min_degree(x_mind.M)
                        rewards_mindegree += x_mind.eliminate_node(node_mind, reduce=True)

                        # baseline2: compute return of random
                        # rewards_random += x_rand.eliminate_node(np.random.randint(low=0, high=x_rand.n), reduce=True)

                        # call actor-critic model

                        node_selected, d_min = x_rl.min_degree(x_rl.M)
                        if not (d_min == 1 or d_min == 0):
                            i += 1
                            action, log_prob, reward, value_current, value_next, x_rl = self.model(
                                x_rl)  # forward propagation,action: node selected, reward: nb edges added
                            self.model.rewards.append(reward)
                            self.model.actions.append(action)
                            self.model.saved_actions.append(SavedAction(log_prob, value_current))
                        else:
                            reward = x_rl.eliminate_node(node_selected, reduce=True)
                    i = 1
                    # while i < depth:
                    #     # gcn-greedy
                    #     node_selected, d_min = x_test.min_degree(x_test.M)
                    #     if not (d_min == 1 or d_min == 0):
                    #         i += 1
                    #
                    #         features = np.ones([x_test.n, 1], dtype=np.float32)
                    #         M_gcn = torch.FloatTensor(x_test.M)
                    #         features = torch.FloatTensor(features)
                    #         M_gcn = utils.to_sparse(M_gcn)  # convert to coo sparse tensor
                    #         if self.use_cuda:
                    #             M_gcn = M_gcn.cuda()
                    #             features = features.cuda()
                    #         probs = self.model.actor(features, M_gcn)
                    #         probs = probs.view(-1)
                    #
                    #         # probs = torch.exp(probs)
                    #         m = Categorical(logits=probs) # logits=probs
                    #         q_gcn_samples = m.sample()
                    #         # q_gcn_samples = np.argmax(np.array(output.detach().cpu().numpy()))
                    #         edges_added = x_test.eliminate_node(q_gcn_samples,
                    #                                             reduce=True)  # eliminate the node and return the number of edges added
                    #         rewards_gcn_greedy += edges_added
                    #     else:
                    #         reward = x_test.eliminate_node(node_selected, reduce=True)

                    rewards_gcn_sto = sum(self.model.rewards)

                    # print('graph {:04d}'.format(n_graphs_proceed), 'epoch {:04d}'.format(epoch),
                    #       'gcn2mind ratio {}'.format(_ratio_gcn2mind),
                    #       'value {}'.format(saved_actions[0].value_current),
                    #       'R {}'.format(returns[0]))
                    # print('graph {:04d}'.format(n_graphs_proceed), 'epoch {:04d}'.format(epoch),
                    #       'gcn2rand ratio {}'.format(_ratio_gcn2rand))

                    _ratio_gcn2mind = rewards_gcn_sto / rewards_mindegree
                    # _ratio_gcn2rand = rewards_gcn / rewards_random
                    gcn_greedy.append(rewards_gcn_greedy)
                    gcn_sto.append(rewards_gcn_sto)
                    mind.append(rewards_mindegree)
                    # rand.append(rewards_random)
                    ratio_gcn2mind.append(_ratio_gcn2mind)
                    # ratio_gcn2rand.append(_ratio_gcn2rand)

                    del self.model.rewards[:]
                    del self.model.actions[:]
                    del self.model.saved_actions[:]

                # n_graphs_proceed += len(X)

            gcn_greedy = np.array(gcn_greedy).reshape(-1)
            gcn_sto = np.array(gcn_sto).reshape(-1)
            mind = np.array(mind).reshape(-1)
            # rand = np.array(rand).reshape(-1)
            ratio_gcn2mind = np.array(ratio_gcn2mind).reshape(-1)
            # ratio_gcn2rand = np.array(ratio_gcn2rand).reshape(-1)

            _ave_gcn_sto = np.sum(gcn_sto) / len(gcn_sto)
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
            ave_gcn_sto.append(_ave_gcn_sto)
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

            # ave_ratio_gcn2rand.append(_ave_ratio_gcn2rand)
            # min_ratio_gcn2rand.append(_min_ratio_gcn2rand)
            # max_ratio_gcn2rand.append(_max_ratio_gcn2rand)

            t_plot = np.array(t).reshape(-1)
            ave_gcn_sto_plot = np.array(ave_gcn_sto).reshape(-1)
            ave_gcn_plot = np.array(ave_gcn).reshape(-1)
            ave_mind_plot = np.array(ave_mind).reshape(-1)
            # ave_rand_plot = np.array(ave_rand).reshape(-1)

            if self.use_cuda:
                plt.clf()
                plt.plot(t_plot, ave_gcn_sto_plot, t_plot, ave_mind_plot) # t_plot, ave_gcn_plot,
                plt.legend(('GNN-RL-epsilon', 'min-degree'), # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
                           loc='upper right')                           # 'GNN-initial', 'GNN-RL', 'min-degree'
                plt.title('RL-MonteCarlo learning curve with pretrain ERG100 (average number of filled edges)')
                plt.ylabel('number of fill-in')
                # plt.draw()
                plt.savefig('./results/acmc01_learning_curve_g2m_number_gcn_logsoftmax_pretrainERG100_UFSM_cuda_with_epsilon0_1_return_test.png')
                plt.clf()
            else:
                plt.clf()
                plt.plot(t_plot, ave_gcn_plot, t_plot, ave_gcn_sto_plot, t_plot, ave_mind_plot)
                plt.legend(('GNN-RL', 'GNN-RL-epsilon', 'min-degree'),
                           loc='upper right')
                plt.title('RL-MonteCarlo learning curve with pretrain ERG100 (average number of filled edges)')
                plt.ylabel('number of fill-in')
                # plt.draw()
                plt.savefig('./results/acmc01_learning_curve_g2m_number_gcn_non_pretrainERG100_with_epsilon05.png')
                plt.clf()

            print('epoch {:04d}'.format(epoch), 'gcn2mind',
                  'min_ratio {}'.format(_min_ratio_gcn2mind),
                  'max_ratio {}'.format(_max_ratio_gcn2mind),
                  'av_ratio {}'.format(_ave_ratio_gcn2mind))
            for name, param in self.model.named_parameters():
                print('parameter name {}'.format(name),
                    'parameter value {}'.format(param.data))
            # print('epoch {:04d}'.format(epoch), 'gcn2rand',
            #       'min_ratio {}'.format(_min_ratio_gcn2rand),
            #       'max_ratio {}'.format(_max_ratio_gcn2rand),
            #       'av_ratio {}'.format(_ave_ratio_gcn2rand),
            #       'nb graph proceeded {}'.format(n_graphs_proceed))

        gcn_greedy = np.array(ave_gcn).reshape(-1)
        ave_ratio_gcn2mind = np.array(ave_ratio_gcn2mind).reshape(-1)
        # ave_ratio_gcn2rand = np.array(ave_ratio_gcn2rand).reshape(-1)

        t = np.arange(0, n_epochs, 1)
        if self.use_cuda:
            plt.clf()
            plt.plot(t, ave_ratio_gcn2mind)
            plt.legend(('GNN-RL/mindegree'),
                       loc='upper right')
            plt.title('RL-MonteCarlo learning curve ratio with pretrain ERG100')
            plt.ylabel('fill-in ratio: gnn model/heuristic')
            plt.savefig('./results/acmc01_learning_curve_g2m_ratio_gcn_logsoftmax_pretrainERG100_UFSM_cuda_with_epsilon0_1_return_test.png')
            plt.clf()
        else:
            plt.clf()
            plt.plot(t, ave_ratio_gcn2mind)
            plt.legend(('GNN-RL/mindegree'),
                       loc='upper right')
            plt.title('RL-MonteCarlo learning curve ratio with pretrain ERG100')
            plt.ylabel('fill-in ratio: gnn model/heuristic')
            plt.savefig('./results/acmc01_learning_curve_g2m_ratio_gcn_non_pretrainERG100_with_epsilon05.png')
            plt.clf()




