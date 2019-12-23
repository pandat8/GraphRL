# import torch
# i = torch.LongTensor([[0, 1, 1],
#                     [2, 0, 2]])
#
# v = torch.LongTensor([1, 1, 1])
#
# a = torch.sparse.LongTensor(i, v, torch.Size([2, 3]))
#
# i2 = torch.LongTensor([[2, 0, 2],
#                     [0, 1, 1]])
#
# v2 = torch.LongTensor([1, 1, 1])
#
# b = torch.sparse.LongTensor(i2, v2, torch.Size([2, 3]))
#
# print(a)
import numpy as np
import torch

l, p = np.loadtxt('./results/logs/log_supervise_gridsearch_mindegree__ER_small_20graphs_train.txt', delimiter=' ', usecols=(7, 9), unpack=True)
print(l)
# with open('./results/logs/log_supervise_gridsearch_mindegree__ER_small_20graphs_train.txt', 'r') as f:
#     d = dict(line.strip().split(' ') for line in f)
# from gcn.models_gcn import GCN_Policy_SelectNode, GCN_Sparse_Policy_SelectNode, GCN_Sparse_Memory_Policy_SelectNode
# actor = GCN_Sparse_Policy_SelectNode(nin=1,
#                               nhidden= 1,
#                               nout=1,
#                               dropout=1,
#                               ) # alpha=args.alpha
#
#
# actor.load_state_dict(torch.load('../results/models/gcn_policy_min_degree_pre_ErgDatasetdense_0.05_epochs20_cuda.pth', map_location=lambda storage, loc: storage))
#
# state_dict = actor.state_dict()
# for name, param in actor.named_parameters():
#     print('parameter name {}'.format(name),
#           'parameter value {}'.format(param.data))
#     print('changing')
#     if name == 'gc1.weight':
#         transformed_param = torch.FloatTensor(
#        [1.00e-02 * 3.6271])
#
#     elif name == 'gc1.bias':
#         transformed_param = torch.FloatTensor([1.7368])
#     elif name == 'gc2.weight':
#         transformed_param = torch.FloatTensor([-0.9479])
#     elif name == 'gc2.bias':
#         transformed_param = torch.FloatTensor(
#        [1.00e-05 *-6.9724])
#     state_dict[name].copy_(transformed_param)
#
# torch.save(actor.state_dict(), '../results/models/gcn_policy_min_degree_pre_ErgDatasetdense_0.05_epochs20_cuda.pth')
#
# actor.load_state_dict(torch.load('../results/models/gcn_policy_min_degree_pre_ErgDatasetdense_0.05_epochs20_cuda.pth', map_location=lambda storage, loc: storage))
#
# for name, param in actor.named_parameters():
#     print('parameter name {}'.format(name),
#           'parameter value {}'.format(param.data))





