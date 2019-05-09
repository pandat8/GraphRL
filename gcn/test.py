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

R = torch.ones([2, 2])
R = R -0.5
R = 1/R
print(R)


