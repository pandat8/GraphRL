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
import torch
import numpy as np
from torch.distributions import Categorical
from data.ergDataset import ErgDataset
from gcn.models_gcn import GCN_Policy_SelectNode, GCN_Sparse_Policy_SelectNode, GCN_Sparse_Memory_Policy_SelectNode
actor = GCN_Sparse_Policy_SelectNode(nin=1,
                              nhidden= 1,
                              nout=1,
                              dropout=1,
                              ) # alpha=args.alpha


actor.load_state_dict(torch.load('../results/models/gcn_policy_min_degree_pre_UFSMDataset_cuda.pth', map_location=lambda storage, loc: storage))

state_dict = actor.state_dict()
for name, param in actor.named_parameters():
    print('parameter name {}'.format(name),
          'parameter value {}'.format(param.data))
    print('changing')
    if name == 'gc1.weight':
        transformed_param = torch.FloatTensor(
       [ 1.00000e-03 *1.6009])
    elif name == 'gc1.bias':
        transformed_param = torch.FloatTensor([3.8930])
    elif name == 'gc2.weight':
        transformed_param = torch.FloatTensor([-0.9935])
    elif name == 'gc2.bias':
        transformed_param = torch.FloatTensor(
       [1.00000e-03 * 2.2737])
    state_dict[name].copy_(transformed_param)

torch.save(actor.state_dict(), '../results/models/gcn_policy_'+'min_degree'+'_pre_'+'UFSMDataset'+'_cuda.pth')

actor.load_state_dict(torch.load('../results/models/gcn_policy_min_degree_pre_UFSMDataset_cuda.pth', map_location=lambda storage, loc: storage))

for name, param in actor.named_parameters():
    print('parameter name {}'.format(name),
          'parameter value {}'.format(param.data))



