import torch
import torch.optim as optm
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from data.graph import Graph
import matplotlib.pyplot as plt


class TrainModel_TD:

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


    def train_and_validate(self, n_epochs, lr_actor, lr_critic, gamma=0.99, use_critic=True):

        self.actor_optim = optm.Adam(self.model.actor.parameters(), lr=lr_actor)

        if use_critic:
            self.critic_optim = optm.Adam(self.model.critic.parameters(), lr=lr_critic)
            self.critic_loss_criterion = torch.nn.MSELoss()
        else:
            baseline = torch.zeros(1)
            if self.use_cuda:
                baseline = baseline.cuda()
        gcn2mind = []
        ratio_gcn2mind = []

        for epoch in range(n_epochs):


            ratio_gcn2rand = []
            n_graphs_proceed = 0
            for X in self.train_loader:
                for x in X:
                    self.model.train()

                    rewards_mindegree = 0  # number of added edges
                    rewards_random = 0
                    x_mind = Graph(x.M)
                    x_rand = Graph(x.M)
                    x = Graph(x.M)
                    n = x.n

                    # loop for training while eliminating a graph iteratively
                    for i in range(n - 2): #x.n - 2

                        # baseline1: compute return of min degree
                        node_mind, d_min = x_mind.min_degree(x_mind.M)
                        rewards_mindegree += x_mind.eliminate_node(node_mind, reduce=True)

                        # baseline2: compute return of random
                        rewards_random += x_rand.eliminate_node(np.random.randint(low=0, high=x_rand.n), reduce=True)

                        # call actor-critic model
                        node_selected, log_prob, r, critic_current, critic_next, x = self.model(x)
                        self.model.rewards.append(r)
                        self.model.actions.append(node_selected)

                        # define baseline here
                        #
                        #

                        # compute reward and advantage term
                        if use_critic:
                            R = r + gamma * critic_next
                            advantage = R - critic_current
                        else:
                            R = r
                            advantage = R - baseline
                        # advantage = r - 500

                        if i==0:
                            print('critic_current {}'.format(critic_current))
                            print('R with 1stepTD {}'.format(R))


                        # if log_prob.data[0] < -1000:
                        #     print(log_prob.data[0])
                        #     log_prob = Variable(torch.FloatTensor([0.]), requires_grad=True)

                        actor_loss = log_prob * advantage.detach() # loss of actor

                        # step training of actor
                        self.actor_optim.zero_grad()
                        actor_loss.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm(self.model.actor.parameters(),
                                                      float(self.max_grad_norm), norm_type=2)
                        self.actor_optim.step()

                        # step training of critic
                        if use_critic:
                            self.critic_optim.zero_grad()
                            R = torch.FloatTensor([R])
                            if self.use_cuda:
                                R = R.cuda()
                            loss_critic = self.critic_loss_criterion(critic_current, R)
                            loss_critic.backward()
                            torch.nn.utils.clip_grad_norm(self.model.critic.parameters(),
                                                          float(self.max_grad_norm), norm_type=2)
                            self.critic_optim.step()
                        else:
                            baseline =  baseline.detach()

                    R = 0
                    for r in self.model.rewards[::-1]:
                        R = r + gamma * R
                    print('return_sampling {}'.format(R))

                    rewards_gcn = sum(self.model.rewards)

                    _ratio_gcn2mind = rewards_gcn / rewards_mindegree
                    _ratio_gcn2rand = rewards_gcn / rewards_random
                    gcn2mind.append(rewards_gcn)
                    ratio_gcn2mind.append(_ratio_gcn2mind)
                    ratio_gcn2rand.append(_ratio_gcn2rand)
                    del self.model.rewards[:]
                    del self.model.actions[:]

                n_graphs_proceed += len(X)

        gcn2mind = np.array(gcn2mind).reshape(-1)
        ratio_gcn2mind = np.array(ratio_gcn2mind).reshape(-1)
        ratio_gcn2rand = np.array(ratio_gcn2rand).reshape(-1)

        total_ratio_gcn2mind = np.sum(ratio_gcn2mind)
        total_ratio_gcn2rand = np.sum(ratio_gcn2rand)

        min_ratio_gcn2mind = np.min(ratio_gcn2mind)
        max_ratio_gcn2mind = np.max(ratio_gcn2mind)
        av_ratio_gcn2mind = total_ratio_gcn2mind / len(ratio_gcn2mind)

        min_ratio_gcn2rand = np.min(ratio_gcn2rand)
        max_ratio_gcn2rand = np.max(ratio_gcn2rand)
        av_ratio_gcn2rand = total_ratio_gcn2rand / len(ratio_gcn2rand)

        print('epoch {:04d}'.format(epoch), 'gcn2mind',
        'min_ratio {}'.format(min_ratio_gcn2mind),
        'max_ratio {}'.format(max_ratio_gcn2mind),
        'av_ratio {}'.format(av_ratio_gcn2mind))
        print('epoch {:04d}'.format(epoch), 'gcn2rand',
              'min_ratio {}'.format(min_ratio_gcn2rand),
              'max_ratio {}'.format(max_ratio_gcn2rand),
              'av_ratio {}'.format(av_ratio_gcn2rand),
              'nb graph proceeded {}'.format(n_graphs_proceed))

        plt.plot(ratio_gcn2mind)
        plt.title('learning curve')
        plt.ylabel('fill-in ratio: gnn model/heuristic')
        plt.savefig('./results/actd_learning_curve_gan.png')
        plt.clf()

        plt.plot(gcn2mind)
        plt.title('learning curve')
        plt.ylabel('nb of fill-in: gnn model')
        plt.savefig('./results/actd_learning_curve2_gan.png')

