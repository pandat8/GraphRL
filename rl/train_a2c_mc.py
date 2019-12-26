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

    def __init__(self, model, heuristic = 'min_degree', train_dataset=None, val_dataset=None, test_dataset=None, weight_d=5e-4, max_grad_norm=2, use_cuda=False):
        self.model = model
        self.heuristic = heuristic
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


    def train_and_validate(self, n_epochs, lr_actor, lr_critic, use_critic, gamma=0.99, density=0.1):
        actor_initial = copy.deepcopy(self.model.actor)

        depth_max = 1000000

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

        total_loss_train = []

        t = []

        val_ave_gcn = []
        train_ave_gcn = []
        # min_gcn = []
        # max_gcn = []

        val_ave_mind = []
        train_ave_mind = []
        # min_mind = []
        # max_mind = []

        val_ave_ratio_gcn2mind = []
        train_ave_ratio_gcn2mind = []
        # min_ratio_gcn2mind = []
        # max_ratio_gcn2mind = []
        # ave_rand = []
        # min_rand = []
        # max_rand = []

        for epoch in range(n_epochs):

            if epoch == 0: # in epoch o, heuristic will be executed and its result acts as a baseline,
                           # in following epochs, heuristic won't be executed anymore (for avoiding unnecessary comuptation).
                val_gcn_greedy = []
                train_gcn_greedy = []
                val_mind = []
                train_mind = []

                av_loss_train = 0  # loss per epochs
                for X in self.train_loader:
                    for x in X:

                        n = x.n

                        self.model.eval()
                        # ratio_gcn2mind = []
                        # ratio_gcn2rand = []

                        train_rewards_mindegree = 0
                        # train_rewards_gcn_greedy = 0
                        # rewards_random = 0
                        # x_mind = Graph(x.M)
                        # x_rand = Graph(x.M)
                        x_model = Graph(x.M)
                        x_mind = Graph(x.M)
                        total_loss_train_1graph = 0

                        # loop for training while eliminating a graph iteratively
                        i = 1
                        depth = np.min([n - 2, depth_max])
                        # depth = n-2
                        rewards_gcn_greedy = np.zeros(1)
                        while (i < depth) and (x_model.n > 2):

                            # baseline1: compute return of min degree
                            # if i % 100 == 0:
                            #     print('iterations {}'.format(i))
                            # node_mind, d_min = x_mind.min_degree(x_mind.M)
                            # rewards_mindegree += x_mind.eliminate_node(node_mind, reduce=True)

                            # baseline2: compute return of random
                            # rewards_random += x_rand.eliminate_node(np.random.randint(low=0, high=x_rand.n), reduce=True)

                            # call actor-critic model

                            if epoch == 0:
                                if self.heuristic == 'min_degree':
                                    node_chosen, d_min = x_mind.min_degree(x_mind.M)
                                elif self.heuristic == 'one_step_greedy':
                                    node_chosen = x_mind.onestep_greedy()
                                # node_mind, d_min = x_mind.min_degree(x_mind.M)
                                train_rewards_mindegree += x_mind.eliminate_node(node_chosen, reduce=True)

                            node_selected, d_min = x_model.min_degree(x_model.M)
                            if not (d_min==1 or d_min == 0):
                                i += 1
                                action, log_prob, reward, value_current, value_next, x_model = self.model(
                                    x_model)  # forward propagation,action: node selected, reward: nb edges added
                                self.model.rewards.append(reward)
                                self.model.actions.append(action)
                                self.model.saved_actions.append(SavedAction(log_prob, value_current))
                            else:
                                reward = x_model.eliminate_node(node_selected, reduce=True)
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

                        returns = returns + 1
                        returns = 1 / returns
                        # returns = -returns.log()

                        # returns = -returns

                        returns = (returns - self.model.epsilon / 530000) / (1 - self.model.epsilon)
                        # returns = returns / (returns.std() + self.eps)

                        # returns = (returns - returns.mean()) / (returns.std() + self.eps)
                        saved_actions = self.model.saved_actions
                        # compute cummulated loss of actor and critic of one graph
                        gamma_t = 1
                        for (log_prob, value_current), R in zip(saved_actions, returns):
                            if use_critic:
                                advantage = R - value_current
                                critic_losses.append(-value_current * advantage)
                                # critic_losses.append(self.critic_loss_criterion(value_current, torch.Tensor([R.detach()])))
                            else:
                                advantage = R - baseline
                            if self.use_cuda:
                                advantage = advantage.cuda()
                            actor_losses.append(
                                -gamma_t * log_prob * advantage.detach())  # the return here is discounted nb of added edges,
                                                                           # hence, if it actually represents loss
                            # gamma_t = gamma_t * gamma

                        # step update of actor
                        actor_loss = torch.stack(actor_losses).sum()
                        total_loss_train_1graph = actor_loss.item()
                        # self.actor_optim.zero_grad()
                        # actor_loss.backward(retain_graph=True)
                        # self.actor_optim.step()
                        # print('epochs {}'.format(epoch), 'loss {}'.format(actor_loss))

                        # step update of critic
                        # if use_critic:
                        #     self.critic_optim.zero_grad()
                        #     critic_closs = torch.stack(critic_losses).sum()
                        #     critic_closs.backward()
                        #     self.critic_optim.step()
                        # else:
                        #     baseline = baseline.detach()

                        train_gcn_greedy.append(sum(self.model.rewards))
                        if epoch == 0:
                            train_mind.append(train_rewards_mindegree)
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

                    av_loss_train += total_loss_train_1graph


                for X in self.val_loader:
                    for x in X:

                        self.model.eval()
                        n = x.n

                        # ratio_gcn2mind = []
                        # ratio_gcn2rand = []
                        val_rewards_mindegree = 0  # number of added edges
                        val_rewards_gcn_greedy = 0
                        # rewards_random = 0
                        x_mind = Graph(x.M)
                        # x_rand = Graph(x.M)
                        x_model = Graph(x.M)

                        # loop for training while eliminating a graph iteratively
                        i = 1
                        depth = np.min([n - 2, depth_max])
                        # depth = n-2
                        while (i < depth) and (x_model.n > 2):

                            # baseline1: compute return of min degree
                            # if i % 100 == 0:
                            #     print('iterations {}'.format(i))
                            if epoch == 0:
                                if self.heuristic == 'min_degree':
                                    node_chosen, d_min = x_mind.min_degree(x_mind.M)
                                elif self.heuristic == 'one_step_greedy':
                                    node_chosen = x_mind.onestep_greedy()
                                # node_mind, d_min = x_mind.min_degree(x_mind.M)
                                val_rewards_mindegree += x_mind.eliminate_node(node_chosen, reduce=True)

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

                                probs = self.model.actor(features, M_gcn)
                                probs = probs.view(-1)
                                # probs = torch.exp(probs)
                                m = Categorical(logits=probs)  # logits=probs
                                q_gcn_samples = m.sample()
                                edges_added = x_model.eliminate_node(q_gcn_samples,
                                                                     reduce=True)  # eliminate the node and return the number of edges added
                                val_rewards_gcn_greedy += edges_added
                            else:
                                reward = x_model.eliminate_node(node_selected, reduce=True)

                        # print('graph {:04d}'.format(n_graphs_proceed), 'epoch {:04d}'.format(epoch),
                        #       'gcn2mind ratio {}'.format(_ratio_gcn2mind),
                        #       'value {}'.format(saved_actions[0].value_current),
                        #       'R {}'.format(returns[0]))
                        # print('graph {:04d}'.format(n_graphs_proceed), 'epoch {:04d}'.format(epoch),
                        #       'gcn2rand ratio {}'.format(_ratio_gcn2rand))

                        # _ratio_gcn2mind = rewards_gcn_greedy / rewards_mindegree
                        # _ratio_gcn2rand = rewards_gcn / rewards_random
                        val_gcn_greedy.append(val_rewards_gcn_greedy)
                        if epoch == 0:
                            val_mind.append(val_rewards_mindegree)
                        # rand.append(rewards_random)
                        # ratio_gcn2mind.append(_ratio_gcn2mind)
                        # ratio_gcn2rand.append(_ratio_gcn2rand)

                    # n_graphs_proceed += len(X)


                # print('epoch {:04d}'.format(epoch), 'gcn2' + self.heuristic,
                #       # 'min_ratio {}'.format(_min_ratio_gcn2mind),
                #       # 'max_ratio {}'.format(_max_ratio_gcn2mind),
                #       ' train av_ratio {}'.format(_train_ave_ratio_gcn2mind),
                #       ' validation av_ratio {}'.format(_val_ave_ratio_gcn2mind))
                for name, param in self.model.named_parameters():
                    print('parameter name {}'.format(name),
                          'parameter value {}'.format(param.data))

            else:
                av_loss_train = 0  # loss per epochs
                # ratio_gcn2rand = []
                n_graphs_proceed = 0
                # for batch_id, sample_batch in enumerate(self.train_loader):
                val_gcn_greedy = []
                train_gcn_greedy = []

                for X in self.train_loader:
                    for x in X:

                        n = x.n

                        self.model.train()
                        # ratio_gcn2mind = []
                        # ratio_gcn2rand = []

                        # train_rewards_mindegree = 0
                        # train_rewards_gcn_greedy = 0
                        # rewards_random = 0

                        # x_rand = Graph(x.M)
                        x_model = Graph(x.M)
                        # x_mind = Graph(x.M)
                        total_loss_train_1graph = 0

                        # loop for training while eliminating a graph iteratively
                        i = 1

                        depth = np.min([n - 2, depth_max])
                        # depth = n-2
                        rewards_gcn_greedy = np.zeros(1)
                        while (i<depth) and (x_model.n > 2):

                            # baseline1: compute return of min degree
                            # if i % 100 == 0:
                            #     print('iterations {}'.format(i))
                            # node_mind, d_min = x_mind.min_degree(x_mind.M)
                            # rewards_mindegree += x_mind.eliminate_node(node_mind, reduce=True)

                            # baseline2: compute return of random
                            # rewards_random += x_rand.eliminate_node(np.random.randint(low=0, high=x_rand.n), reduce=True)

                            # call actor-critic model

                            # if epoch==0:
                            #     if self.heuristic == 'min_degree':
                            #         node_chosen, d_min = x_mind.min_degree(x_mind.M)
                            #     elif self.heuristic == 'one_step_greedy':
                            #         node_chosen, d_min = x_mind.onestep_greedy()
                            #     # node_mind, d_min = x_mind.min_degree(x_mind.M)
                            #     train_rewards_mindegree += x_mind.eliminate_node(node_chosen, reduce=True)

                            node_selected, d_min = x_model.min_degree(x_model.M)
                            if not (d_min==1 or d_min ==0 ):
                                i += 1
                                action, log_prob, reward, value_current, value_next, x_model = self.model(x_model) # forward propagation,action: node selected, reward: nb edges added
                                self.model.rewards.append(reward)
                                self.model.actions.append(action)
                                self.model.saved_actions.append(SavedAction(log_prob, value_current))
                            else:
                                reward = x_model.eliminate_node(node_selected, reduce=True)
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
                        # returns = -returns.log()

                        # returns = -returns

                        returns = (returns - self.model.epsilon/ 530000) / (1 - self.model.epsilon)
                        # returns = returns / (returns.std() + self.eps)

                        # returns = (returns - returns.mean()) / (returns.std() + self.eps)

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
                            # gamma_t = gamma_t*gamma

                        # step update of actor
                        actor_loss = torch.stack(actor_losses).sum()
                        total_loss_train_1graph = actor_loss.item()
                        self.actor_optim.zero_grad()
                        actor_loss.backward(retain_graph=True)
                        self.actor_optim.step()
                        # print('epochs {}'.format(epoch), 'loss {}'.format(actor_loss))

                        # step update of critic
                        if use_critic:
                            self.critic_optim.zero_grad()
                            critic_closs = torch.stack(critic_losses).sum()
                            critic_closs.backward()
                            self.critic_optim.step()
                        else:
                            baseline = baseline.detach()


                        train_gcn_greedy.append(sum(self.model.rewards))
                        # if epoch==0:
                        #     train_mind.append(train_rewards_mindegree)
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

                    av_loss_train += total_loss_train_1graph



                for X in self.val_loader:
                    for x in X:

                        self.model.eval()
                        n = x.n

                        # ratio_gcn2mind = []
                        # ratio_gcn2rand = []
                        # val_rewards_mindegree = 0  # number of added edges
                        val_rewards_gcn_greedy = 0
                        # rewards_random = 0
                        x_mind = Graph(x.M)
                        # x_rand = Graph(x.M)
                        x_model = Graph(x.M)

                        # loop for training while eliminating a graph iteratively
                        i = 1
                        depth = np.min([n - 2, depth_max])
                        # depth = n-2
                        while (i < depth) and (x_model.n > 2):

                            # if epoch==0:
                            #     if self.heuristic == 'min_degree':
                            #         node_chosen, d_min = x_mind.min_degree(x_mind.M)
                            #     elif self.heuristic == 'one_step_greedy':
                            #         node_chosen, d_min = x_mind.onestep_greedy()
                            #     # node_mind, d_min = x_mind.min_degree(x_mind.M)
                            #     val_rewards_mindegree += x_mind.eliminate_node(node_chosen, reduce=True)

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

                                probs = self.model.actor(features, M_gcn)
                                probs = probs.view(-1)
                                # probs = torch.exp(probs)
                                m = Categorical(logits=probs) # logits=probs
                                q_gcn_samples = m.sample()
                                edges_added = x_model.eliminate_node(q_gcn_samples,
                                                                    reduce=True)  # eliminate the node and return the number of edges added
                                val_rewards_gcn_greedy += edges_added
                            else:
                                reward = x_model.eliminate_node(node_selected, reduce=True)

                        # print('graph {:04d}'.format(n_graphs_proceed), 'epoch {:04d}'.format(epoch),
                        #       'gcn2mind ratio {}'.format(_ratio_gcn2mind),
                        #       'value {}'.format(saved_actions[0].value_current),
                        #       'R {}'.format(returns[0]))
                        # print('graph {:04d}'.format(n_graphs_proceed), 'epoch {:04d}'.format(epoch),
                        #       'gcn2rand ratio {}'.format(_ratio_gcn2rand))

                        # _ratio_gcn2mind = rewards_gcn_greedy / rewards_mindegree
                        # _ratio_gcn2rand = rewards_gcn / rewards_random
                        val_gcn_greedy.append(val_rewards_gcn_greedy)
                        # if epoch==0:
                        #     val_mind.append(val_rewards_mindegree)
                        # rand.append(rewards_random)
                        # ratio_gcn2mind.append(_ratio_gcn2mind)
                        # ratio_gcn2rand.append(_ratio_gcn2rand)

                    # n_graphs_proceed += len(X)

            train_gcn_greedy = np.array(train_gcn_greedy).reshape(-1)
            val_gcn_greedy = np.array(val_gcn_greedy).reshape(-1)
            _val_ave_gcn = np.sum(val_gcn_greedy) / len(val_gcn_greedy)
            _train_ave_gcn = np.sum(train_gcn_greedy) / len(train_gcn_greedy)

            if epoch == 0:
                val_mind = np.array(val_mind).reshape(-1)
                train_mind = np.array(train_mind).reshape(-1)
                _val_ave_mind = np.sum(val_mind) / len(val_mind)
                _train_ave_mind = np.sum(train_mind) / len(train_mind)




            t.append(epoch+1)
            val_ave_gcn.append(_val_ave_gcn)
            train_ave_gcn.append(_train_ave_gcn)
            val_ave_mind.append(_val_ave_mind)
            train_ave_mind.append(_train_ave_mind)

            # print('epochs {}'.format(epoch),'loss {}'.format(av_loss_train) )
            total_loss_train.append(av_loss_train)

            # print('epochs {}'.format(epoch),
            #       'loss {}'.format(av_loss_train),
            #       'train '+ self.heuristic + 'performance {}'.format(_train_ave_mind),
            #       'train gcn performance {}'.format(_train_ave_gcn),
            #       'val ' + self.heuristic + 'performance {}'.format(_val_ave_mind),
            #       'val gcn performance {}'.format(_val_ave_gcn),
            #
            #       )


            # _val_ave_ratio_gcn2mind = _val_ave_gcn / _val_ave_mind
            # _train_ave_ratio_gcn2mind = _train_ave_gcn / _train_ave_mind
            # val_ave_ratio_gcn2mind.append(_val_ave_ratio_gcn2mind)
            # train_ave_ratio_gcn2mind.append(_train_ave_ratio_gcn2mind)


            t_plot = np.array(t).reshape(-1)

            total_loss_train_np = np.array(total_loss_train).reshape(-1)

            val_ave_gcn_np = np.array(val_ave_gcn).reshape(-1)
            train_ave_gcn_np = np.array(train_ave_gcn).reshape(-1)
            val_ave_mind_np = np.array(val_ave_mind).reshape(-1)
            train_ave_mind_np = np.array(train_ave_mind).reshape(-1)

            plt.clf()
            plt.plot(t_plot, train_ave_gcn_np, t_plot, train_ave_mind_np)
            plt.legend(('GNN-RL-epsilon', self.heuristic),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
                       loc='upper right')  # 'GNN-initial', 'GNN-RL', 'min-degree'
            plt.title(
                'RL-MonteCarlo performance curve with pretrain trainDataset ' + self.train_dataset.__class__.__name__ + ' (average number of filled edges)')
            plt.ylabel('number of fill-in')
            # plt.draw()
            plt.savefig(
                './results/rl/rmc/hyper_lractor_acmc_power-1_r_' + str(
                    lr_actor) + '_epsilon_' + str(self.model.epsilon.numpy()) + '_' + self.heuristic + '_curve_g2m_number_gcn_logsoftmax_no_pretrain_train_' + self.train_dataset.__class__.__name__ + '_unlim_depth_prune_cuda' + str(
                    self.use_cuda) + '_without_epsilon_return_-mean-eps02.png')
            plt.clf()

            plt.clf()
            plt.plot(t_plot, val_ave_gcn_np, t_plot, val_ave_mind_np)
            plt.legend(('GNN-RL-epsilon', self.heuristic),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
                       loc='upper right')  # 'GNN-initial', 'GNN-RL', 'min-degree'
            plt.title(
                'RL-MonteCarlo performance curve with pretrain validationDataset' + self.train_dataset.__class__.__name__ + ' (average number of filled edges)')
            plt.ylabel('number of fill-in')
            # plt.draw()
            plt.savefig(
                './results/rl/rmc/hyper_lractor_acmc_power-1_r_' + str(
                    lr_actor) + '_epsilon_' + str(self.model.epsilon.numpy())  + '_' + self.heuristic + '_curve_g2m_number_gcn_logsoftmax_no_pretrain_val_' + self.train_dataset.__class__.__name__ + '_unlim_depth_prune_cuda' + str(
                    self.use_cuda) + '_return_-mean-eps02.png')
            plt.clf()

            plt.clf()
            plt.plot(t_plot, total_loss_train_np)
            plt.legend(('GCN-PolicyGradient-loss'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
                       loc='upper right')  # 'GNN-initial', 'GNN-RL', 'min-degree'
            plt.title(
                'RL-MonteCarlo training loss curve with pretrain' + self.train_dataset.__class__.__name__ + ' (average number of filled edges)')
            plt.ylabel('number of fill-in')
            # plt.draw()
            plt.savefig(
                './results/rl/rmc/hyper_lractor_acmc_power-1_r_' + str(
                    lr_actor) + '_epsilon_' + str(self.model.epsilon.numpy()) + '_' + self.heuristic + '_loss_curve_g2m_number_gcn_logsoftmax_no_pretrain_val_' + self.train_dataset.__class__.__name__ + '_umlim_depth_prune_cuda' + str(
                    self.use_cuda) + '_return_-mean-eps02.png')
            plt.clf()

            # plt.clf()
            # plt.plot(t_plot, train_ave_gcn_np, t_plot, train_ave_mind_np)
            # plt.legend(('GNN-RL-epsilon', self.heuristic),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
            #            loc='upper right')  # 'GNN-initial', 'GNN-RL', 'min-degree'
            # plt.title(
            #     'RL-MonteCarlo performance curve with pretrain trainDataset ' + self.train_dataset.__class__.__name__ + ' (average number of filled edges)')
            # plt.ylabel('number of fill-in')
            # # plt.draw()
            # plt.savefig(
            #     './results/rl/hypertraining_lractor_acmc_' + str(
            #         lr_actor) + '_epsilon_'+ str(self.model.epsilon) + '_' + self.heuristic + '_curve_g2m_number_gcn_logsoftmax__with_pretrain_train_' + self.train_dataset.__class__.__name__ + 'fulldepth_cuda' + str(
            #         self.use_cuda) + '_without_epsilon_return_test.png')
            # plt.clf()
            #
            # plt.clf()
            # plt.plot(t_plot, val_ave_gcn_np, t_plot, val_ave_mind_np)
            # plt.legend(('GNN-RL-epsilon', self.heuristic),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
            #            loc='upper right')  # 'GNN-initial', 'GNN-RL', 'min-degree'
            # plt.title(
            #     'RL-MonteCarlo performance curve with pretrain validationDataset' + self.train_dataset.__class__.__name__ + ' (average number of filled edges)')
            # plt.ylabel('number of fill-in')
            # # plt.draw()
            # plt.savefig(
            #     './results/rl/hypertraining_lractor_acmc_' + str(
            #         lr_actor) + '_epsilon_'+ str(self.model.epsilon) +'_'+ self.heuristic + '_curve_g2m_number_gcn_logsoftmax_with_pretrain_val_' + self.train_dataset.__class__.__name__ + 'fulldepth_cuda' + str(
            #         self.use_cuda) + '_without_epsilon_return_test.png')
            # plt.clf()
            #
            # plt.clf()
            # plt.plot(t_plot, total_loss_train_np)
            # plt.legend(('GCN-PolicyGradient-loss'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
            #            loc='upper right')  # 'GNN-initial', 'GNN-RL', 'min-degree'
            # plt.title(
            #     'RL-MonteCarlo training loss curve with pretrain' + self.train_dataset.__class__.__name__ + ' (average number of filled edges)')
            # plt.ylabel('number of fill-in')
            # # plt.draw()
            # plt.savefig(
            #     './results/rl/hypertraining_lractor_acmc_' + str(
            #         lr_actor) + '_epsilon_'+ str(self.model.epsilon) +'_' + self.heuristic + '_loss_curve_g2m_number_gcn_logsoftmax_with_pretrain_val_' + self.train_dataset.__class__.__name__ + 'fulldepth_cuda' + str(
            #         self.use_cuda) + '_without_epsilon_return_test.png')
            # plt.clf()

            # print('epoch {:04d}'.format(epoch+1), 'gcn2' + self.heuristic,
            #       # 'min_ratio {}'.format(_min_ratio_gcn2mind),
            #       # 'max_ratio {}'.format(_max_ratio_gcn2mind),
            #       ' train av_ratio {}'.format(_train_ave_ratio_gcn2mind),
            #       ' validation av_ratio {}'.format(_val_ave_ratio_gcn2mind))
            # for name, param in self.model.named_parameters():
            #     print('parameter name {}'.format(name),
            #           'parameter value {}'.format(param.data))
