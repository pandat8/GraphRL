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
R = 0.01
a = 'abc'
dataset = ErgDataset
train_dataset = dataset(100, 1)
dataset_name = dataset.__name__
print(a+str(R)+ dataset_name)


