from torch.utils.data import Dataset, DataLoader
from data.ergDataset import ErgDataset
from data.SSMCDataset import SSMCDataset
from data.UFSMDataset import UFSMDataset
from data.UFSMDataset_Demo import UFSMDataset_Demo
from data.graph import Graph

from rl.model_a2c import Model_A2C_Sparse
from rl.train_a2c_td import TrainModel_TD
from rl.train_a2c_mc import TrainModel_MC
from gcn.models_gcn import GCN_Policy_SelectNode, GCN_Sparse_Policy_SelectNode, GCN_Sparse_Memory_Policy_SelectNode, GAN, GNN_GAN
from gcn.models_gcn import GCN_Value, GCN_Sparse_Value
from supervised.train_supervised_learning import Train_SupervisedLearning

import sys
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt



# Training argument setting
parser = argparse.ArgumentParser()
parser.add_argument('--nocuda', action= 'store_true', default=False, help='Disable Cuda')
parser.add_argument('--novalidation', action= 'store_true', default=False, help='Disable validation')
parser.add_argument('--seed', type=int, default=50, help='Radom seed')
parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
parser.add_argument('--pretrain_epochs', type=int, default=3, help='Training epochs')
parser.add_argument('--lr_actor', type=float, default= 0.001, help='Learning rate of actor')
parser.add_argument('--lr_critic', type=float, default= 0.001, help='Learning rate of critic')
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--dhidden', type=int, default=1, help='Dimension of hidden features')
parser.add_argument('--dinput', type=int, default=1, help='Dimension of input features')
parser.add_argument('--doutput', type=int, default=1, help='Dimension of output features')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout Rate')
parser.add_argument('--alpha', type=float, default=0.2, help='Aplha')
parser.add_argument('--nnode', type=int, default=300, help='Number of node per graph')
parser.add_argument('--ngraph', type=int, default=20, help='Number of graph per dataset')
parser.add_argument('--p', type=int, default=0.01, help='probiblity of edges')
parser.add_argument('--nnode_test', type=int, default=10, help='Number of node per graph for test')
parser.add_argument('--ngraph_test', type=int, default=1, help='Number of graph for test dataset')
parser.add_argument('--use_critic', type=bool, default=False, help='Enable critic')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.cuda = not args.nocuda and torch.cuda.is_available()

if args.cuda:
   torch.cuda.manual_seed(args.seed)


# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
# val_loader = DataLoader(val_dataset, batch_size=1,  shuffle=True, collate_fn=lambda x: x)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
# test_loader = DataLoader(test_set, batch_size=1, collate_fn=lambda x: x)
# build the GCN model
# actor = GCN_Sparse_Policy_SelectNode(nin=args.dinput,
#                               nhidden= args.dhidden,
#                               nout=args.doutput,
#                               dropout=args.dropout,
#                               ) # alpha=args.alpha
#
# reset nb of epochs of supervised pretraining to load the pre-trained model
#
# if args.cuda:
    # actor.load_state_dict(
    #     torch.load('./results/models/gcn_policy_' + heuristic + '_pre_' + dataset.__name__ +str(args.nnode)+ 'dense_' + str(
    #                args.p) + '_epochs' + str(args.pretrain_epochs) + '_cuda.pth'))
    # actor.load_state_dict(torch.load('./results/models/gcn_policy_one_step_greedy_pre_UFSMDataset_epochs30_cuda.pth'))
    # actor.cuda()
# else:
#     actor.load_state_dict(torch.load('./results/models/gcn_memory_policy_min_degree_pre_erg100.pthh'))

epoch = 0
# # test pretrained policy model
# train_sl = Train_SupervisedLearning(model=actor, test_dataset=test_dataset, use_cuda = args.cuda)
# features = np.ones([args.nnode,args.dinput], dtype=np.float32) # initialize the features for training set
# print('test stated')
# time_start = time.time()
# ratio, av_ratio, max_ratio, min_ratio, ratio_g2r, av_ratio_g2r, max_ratio_g2r, min_ratio_g2r = train_sl.test()
# time_end = time.time()
# print('test finished')
# print('Test time: {:.4f}'.format(time_end-time_start))
#
# # save/plot test results
# if args.cuda:
#     text_file = open("test/results/pretrain_mindegree_gcn_memory_ERG100_cuda.txt", "w")
# else:
#     text_file = open("test/results/pretrain_mindegree_gcn_memory_ERG100.txt", "w")
# text_file.write('\n Pretrained-model Test result: test_gcn_learn_mindegree\n')
# text_file.write('DataSet: GraphDataset\n')
# text_file.write('average ratio {:.4f} \n'.format(av_ratio))
# text_file.write('max ratio {:.4f}\n'.format(max_ratio))
# text_file.write('min ratio {:.4f}\n'.format(min_ratio))
# text_file.write('average ratio graph2random {:.4f}\n'.format(av_ratio_g2r))
# text_file.write('max ratio graph2random {:.4f}\n'.format(max_ratio_g2r))
# text_file.write('min ratio graph2random {:.4f}\n'.format(min_ratio_g2r))
# text_file.close()
#
# if args.cuda:
#     plt.switch_backend('agg')
#     plt.hist(ratio, bins=32)
#     plt.title('histogram: graph2mindegree ratio of Erdos-Renyi graph')
#     plt.savefig('./test/results/histogram_gnn2mindegree_gcn_memory_erg100_cuda.png')
#     plt.clf()
#     #
#     plt.hist(ratio_g2r, bins=32)
#     plt.title('graph2random_ratio_CrossEntropy Erdos-Renyi graph')
#     plt.savefig('./test/results/histogram_gnn2random_gcn_memory_erg100_cuda.png')
#     plt.clf()
# else:
#     plt.hist(ratio, bins= 32)
#     plt.title('histogram: graph2mindegree ratio of Erdos-Renyi graph')
#     plt.savefig('./test/results/histogram_gnn2mindegree_gcn_memory_erg100.png')
#     plt.clf()
#     #
#     plt.hist(ratio_g2r, bins= 32)
#     plt.title('graph2random_ratio_CrossEntropy Erdos-Renyi graph')
#     plt.savefig('./test/results/histogram_gnn2random_gcn_memory_erg100.png')
#     plt.clf()
#     #

# option of critic
critic = None

if args.use_critic:
    critic = GCN_Sparse_Value(nin=args.dinput,
                              nhidden=args.dhidden,
                              nout=args.doutput,
                              dropout=args.dropout,
                              ) # alpha=args.alpha


heuristic = 'min_degree' # 'min_degree' 'one_step_greedy'

# load data and pre-process
dataset = UFSMDataset_Demo
dataset_name = dataset.__name__[0:11]


# train RL-model

print('Supervised Training started')
print('heuristic: '+heuristic,
      'actor learning rate: {}'.format(args.lr_actor),
      'epochs: {}'.format(args.epochs),
      'DataSet: '+dataset.__name__+'\n')
eps = [0, 0.001, 0.01 ,0.02, 0.05, 0.1, 0.2, 0.5 ]

# lr = [0.00001, 0.0001, 0.001, 0.01, 0.1]

# lr = [0.00001, 0.0001, 0.001, ]
lr = [0.1, 0.01, 0.001]
# lr = [0.00001, 0.0001, 0.001,0.1]
time_start = time.time()

for i in range(len(lr)):

    # actor = GCN_Sparse_Policy_SelectNode(nin=args.dinput,
    #                                      nhidden=args.dhidden,
    #                                      nout=args.doutput,
    #                                      dropout=args.dropout,
    #                                      )  # alpha=args.alpha

    actor = GAN(nin=args.dinput,
                nhidden=args.dhidden,
                nout=args.doutput,
                dropout=args.dropout,
                alpha=args.alpha
                )  # alpha=args.alpha

    if dataset_name == 'UFSMDataset':
        test_dataset = dataset(start=24, end=26)
        train_dataset = dataset(start=18, end=19)
        # val_dataset = dataset(args.nnode_test, args.ngraph_test, args.p)
        val_dataset = dataset(start=19, end=19)
    elif dataset_name == 'ErgDataset':
        train_dataset = dataset(args.nnode, args.ngraph, args.p)
        val_dataset = dataset(args.nnode, args.ngraph, args.p)
        test_dataset = dataset(args.nnode_test, args.ngraph_test, args.p)

    if args.cuda:
        # actor.load_state_dict(
        #     torch.load('./results/models/gcn_policy_' + heuristic + '_pre_' + dataset.__name__ + str(
        #         args.nnode) + 'dense_' + str(
        #         args.p) + '_epochs' + str(args.pretrain_epochs) + '_cuda.pth'))
        # actor.load_state_dict(torch.load('./results/models/gcn_policy_one_step_greedy_pre_UFSMDataset_epochs30_cuda.pth'))
        actor.cuda()
    model_a2c = Model_A2C_Sparse(actor=actor,
                                 epsilon=0.0,  # non-pretrain:0.02 #pretrain:0.0
                                 use_critic=args.use_critic,
                                 use_cuda=args.cuda,
                                 critic=critic)
    if args.cuda:
        model_a2c.cuda()

    train_a2c = TrainModel_MC(model_a2c,
                              heuristic=heuristic,
                              train_dataset=train_dataset,
                              val_dataset=val_dataset,
                              weight_d=args.wd,
                              use_cuda=args.cuda)
    # args.lr_actor = lr[i]
    train_a2c.train_and_validate(n_epochs=args.epochs,
                             lr_actor=lr[i], # args.lr_actor
                             lr_critic=args.lr_critic,
                             use_critic = args.use_critic,
                             density = args.p
                             )
print('Training finished')
print('Training time: {:.4f}'.format(time.time()-time_start))





