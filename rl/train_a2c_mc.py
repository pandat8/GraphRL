import torch
import torch.optim as optm
import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from data.graph import Graph
from collections import namedtuple

SavedAction = namedtuple('SavedAction', ['log_prob', 'value_current'])

# Mont Carlo methods
class TrainModel_MC:

    def __init__(self, model, train_dataset, val_dataset, max_grad_norm=2, use_cuda=False):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.max_grad_norm = max_grad_norm
        self.use_cuda = use_cuda

        self.train_loader = DataLoader(train_dataset, shuffle=True, num_workers=1, batch_size=1, collate_fn=lambda x: x)
        self.val_loader = DataLoader(val_dataset, shuffle=True, num_workers=1, batch_size=1, collate_fn=lambda x: x)

        self.epochs = 0
        self.beta = 0.9
        self.eps = np.finfo(np.float32).eps.item()


    def train_and_validate(self, n_epochs, lr_actor, lr_critic, gamma=0.99, use_critic=True):

        self.actor_optim = optm.Adam(self.model.actor.parameters(), lr=lr_actor)

        print(use_critic)
        if use_critic:
            self.critic_optim = optm.Adam(self.model.critic.parameters(), lr=lr_critic)
            self.critic_loss_criterion = torch.nn.MSELoss()
        else:
            baseline = torch.zeros(1)
            if self.use_cuda:
                baseline = baseline.cuda()

        for epoch in range(1):

            n_graphs_proceed = 0
            for X in self.train_loader:
                for x in X:

                    self.model.train()
                    ratio_gcn2mind = []
                    ratio_gcn2rand = []

                    for epoch in range(n_epochs):

                        rewards_mindegree = 0  # number of added edges
                        rewards_random = 0
                        x_mind = Graph(x.M)
                        x_rand = Graph(x.M)
                        x_rl = Graph(x.M)

                        # loop for training while eliminating a graph iteratively
                        for i in range(x.n - 2):

                            # baseline1: compute return of min degree
                            if i % 100 == 0:
                                print('iterations {}'.format(i))
                            node_mind, d_min = x_mind.min_degree(x_mind.M)
                            rewards_mindegree += x_mind.eliminate_node(node_mind, reduce=True)

                            # baseline2: compute return of random
                            rewards_random += x_rand.eliminate_node(np.random.randint(low=0, high=x_rand.n), reduce=True)

                            # call actor-critic model

                            action, log_prob, reward, value_current, value_next, x_rl = self.model(x_rl) # forward propagation,action: node selected, reward: nb edges added
                            self.model.rewards.append(reward)
                            self.model.actions.append(action)
                            self.model.saved_actions.append(SavedAction(log_prob, value_current))

                        R = 0
                        actor_losses = []
                        critic_losses = []
                        returns = []

                        # compute sampled return for each step
                        for r in self.model.rewards[::-1]:
                            R = r + gamma * R
                            returns.insert(0, R)
                        returns = torch.tensor(returns)
                        returns = (returns - returns.mean()) / (returns.std() + self.eps)
                        saved_actions = self.model.saved_actions
                        # compute cummulated loss of actor and critic of one graph
                        for (log_prob, value_current), R in zip(saved_actions, returns):
                            if use_critic:
                                advantage = R - value_current
                                critic_losses.append(-value_current* advantage)
                                # critic_losses.append(self.critic_loss_criterion(value_current, torch.Tensor([R.detach()])))
                            else:
                                advantage = R - baseline
                            actor_losses.append(log_prob * advantage.detach())  # the return here is discounted nb of added edges,
                                                                       # hence, it actually represents loss
                        # step update of actor
                        self.actor_optim.zero_grad()
                        actor_loss = torch.stack(actor_losses).sum()
                        actor_loss.backward(retain_graph=True)
                        self.actor_optim.step()

                        # step update of critic
                        if use_critic:
                            self.critic_optim.zero_grad()
                            critic_closs = torch.stack(critic_losses).sum()
                            critic_closs.backward()
                            self.critic_optim.step()
                        else:
                            baseline = baseline.detach()

                        rewards_gcn = sum(self.model.rewards)

                        _ratio_gcn2mind = rewards_gcn / rewards_mindegree
                        _ratio_gcn2rand = rewards_gcn / rewards_random

                        print('graph {:04d}'.format(n_graphs_proceed), 'epoch {:04d}'.format(epoch),
                              'gcn2mind ratio {}'.format(_ratio_gcn2mind),
                              'value {}'.format(saved_actions[0].value_current),
                              'R {}'.format(returns[0]))
                        print('graph {:04d}'.format(n_graphs_proceed), 'epoch {:04d}'.format(epoch),
                              'gcn2rand ratio {}'.format(_ratio_gcn2rand))

                        ratio_gcn2mind.append(_ratio_gcn2mind)
                        ratio_gcn2rand.append(_ratio_gcn2rand)
                        del self.model.rewards[:]
                        del self.model.actions[:]
                        del self.model.saved_actions[:]

                    ratio_gcn2mind = np.array(ratio_gcn2mind).reshape(-1)
                    ratio_gcn2rand = np.array(ratio_gcn2rand).reshape(-1)

                    min_ratio_gcn2mind = np.min(ratio_gcn2mind)
                    max_ratio_gcn2mind = np.max(ratio_gcn2mind)
                    av_ratio_gcn2mind = np.sum(ratio_gcn2mind)/ n_epochs

                    min_ratio_gcn2rand = np.min(ratio_gcn2rand)
                    max_ratio_gcn2rand = np.max(ratio_gcn2rand)
                    av_ratio_gcn2rand = np.sum(ratio_gcn2rand) / n_epochs

                    print('graph {:04d}'.format(n_graphs_proceed), 'gcn2mind{:04d}',
                          'min_ratio {}'.format(min_ratio_gcn2mind),
                          'max_ratio {}'.format(max_ratio_gcn2mind),
                          'av_ratio {}'.format(av_ratio_gcn2mind))
                    print('graph {:04d}'.format(n_graphs_proceed), 'gcn2rand{:04d}',
                          'min_ratio {}'.format(min_ratio_gcn2rand),
                          'max_ratio {}'.format(max_ratio_gcn2rand),
                          'av_ratio {}'.format(av_ratio_gcn2rand),
                          'nb graph proceeded {}'.format(n_graphs_proceed))

                n_graphs_proceed += len(X)

            # ratio_gcn2mind = np.array(ratio_gcn2mind).reshape(-1)
            # ratio_gcn2rand = np.array(ratio_gcn2rand).reshape(-1)
            #
            # total_ratio_gcn2mind = np.sum(ratio_gcn2mind)
            # total_ratio_gcn2rand = np.sum(ratio_gcn2rand)
            #
            # min_ratio_gcn2mind = np.min(ratio_gcn2mind)
            # max_ratio_gcn2mind = np.max(ratio_gcn2mind)
            # av_ratio_gcn2mind = total_ratio_gcn2mind / n_graphs_proceed
            #
            # min_ratio_gcn2rand = np.min(ratio_gcn2rand)
            # max_ratio_gcn2rand = np.max(ratio_gcn2rand)
            # av_ratio_gcn2rand = total_ratio_gcn2rand / n_graphs_proceed
            #
            # print('epoch {:04d}'.format(epoch), 'gcn2mind{:04d}',
            # 'min_ratio {}'.format(min_ratio_gcn2mind),
            # 'max_ratio {}'.format(max_ratio_gcn2mind),
            # 'av_ratio {}'.format(av_ratio_gcn2mind))
            # print('epoch {:04d}'.format(epoch), 'gcn2rand{:04d}',
            #       'min_ratio {}'.format(min_ratio_gcn2rand),
            #       'max_ratio {}'.format(max_ratio_gcn2rand),
            #       'av_ratio {}'.format(av_ratio_gcn2rand),
            #       'nb graph proceeded {}'.format(n_graphs_proceed))
