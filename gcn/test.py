import scipy.io as spio
import numpy as np
import torch
from data.graph import Graph

# def func_a(*args):
#     temp = tuple(arg * 2 for arg in args)
#     return temp if len(temp) > 1 else temp[0]
#
# if __name__ == '__main__':
#     a, b, c, d = func_a(1, 2, 3, 4)
#     print(a, b, c, d)
#     e = func_a('Hello')
#     print(e)
import numpy as np
# list = []
# list.append(1)
# list.append(2)

# y= torch.nonzero(torch.FloatTensor([1.1]))
# indices = y.t()
# print(indices)
# x = torch.FloatTensor([[1,2],[1,2]])
# print(x.size())
#
# x = np.array([0.1,0.0,0])
# print(x.astype(bool))

from data.UFSMDataset import UFSMDataset
# dataset=[]
# for i in range(18, 25):
#     folder = 'data/UFSMcollection/c-' + str(i) + '.mtx'
#     fxm3_6 = spio.mmread(folder)
#     fxm3_6 = fxm3_6.todense()
#     np.fill_diagonal(fxm3_6, 0)
#     fxm3_6 = fxm3_6.astype(np.uint8)
#     g = Graph(fxm3_6)
#     dataset.append(g)
#
# print(dataset)
# x = torch.FloatTensor([[1]])
# y = torch.nonzero((x))
#
# if list(y):
#     indices = y.t()
#     values = x[indices[0], indices[1]]  #
#     sparse = torch.sparse.FloatTensor(indices, values, x.size())
#
# else:
#     sparse = torch.sparse.FloatTensor(x.size())

# i = torch.LongTensor([[0, 1, 1],
#                           [2, 0, 2]])
# v = torch.FloatTensor([3, 4, 5])
# x = torch.sparse.FloatTensor(i, v, torch.Size([2,3]))
# y = x._indices()
# if y.numel():
#     print(y.numel())
