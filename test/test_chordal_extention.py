import numpy as np
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from data.ergDataset import ErgDataset
from gcn.models_gcn import GCN_Policy_SelectNode

# Training argument setting
parser = argparse.ArgumentParser()
parser.add_argument('--nocuda', action= 'store_true', default=False, help='Disable Cuda')
parser.add_argument('--novalidation', action= 'store_true', default=False, help='Disable validation')
parser.add_argument('--seed', type=int, default=40, help='Radom seed')
parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
parser.add_argument('--lr', type=float, default= 0.01, help='Learning rate')
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--nhidden', type=int, default=1, help='Number of hidden features')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout Rate')
parser.add_argument('--alpha', type=float, default=0.2, help='Aplha')
parser.add_argument('--nnode', type=int, default=20, help='Number of node per graph')
parser.add_argument('--ngraph', type=int, default=100, help='Number of graph per dataset')
parser.add_argument('--nnode_test', type=int, default=20, help='Number of node per graph for test')
parser.add_argument('--ngraph_test', type=int, default=2, help='Number of graph for test dataset')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.cuda = not args.nocuda and torch.cuda.is_available()

if args.cuda:
   torch.cuda.manual_seed(args.seed)


# load data and pre-process
train_set = ErgDataset(args.nnode, args.ngraph)
val_set = ErgDataset(args.nnode, args.ngraph)
test_set = ErgDataset(args.nnode_test, args.ngraph_test)

train_loader = DataLoader(train_set, batch_size=1, collate_fn=lambda x: x)
val_loader = DataLoader(val_set, batch_size=1, collate_fn=lambda x: x)
test_loader = DataLoader(test_set, batch_size=1, collate_fn=lambda x: x)


def evaluate(data_loader, is_cuda=args.cuda, validation = True):
    """
    Evaluation function
    :param model: network model
    :param data_loader: dataset loader depending on validation or test
    :param features: initial feature vector of graph
    :param is_cuda:
    :param validation: True if validation(by default), False if test
    :return: averaged loss per graph
    """
    total_acc = 0
    total_loss = 0
    n_graphs_proceed = 0
    for X in data_loader:
        for x in X:
            M = x.M
            q = x.elimination_ordering()
            x.chordal_extension(q)
            num_e = x.num_e
            M_ex = x.M_ex
            # num_e = torch.IntTensor(x.num_e)
            if is_cuda:
                m = m.cuda()
                features = features.cuda()
                degree = degree.cuda()
            print('epoch {:04d}'.format(epoch),
            'original graph {}'.format(M))
            print('epoch {:04d}'.format(epoch),
            'elimination ordering {}'.format(q))

            print('epoch {:04d}'.format(epoch),
            'number of edges {}'.format(num_e))

            print('epoch {:04d}'.format(epoch),
            'chordal graph {}'.format(M_ex))


epoch = 0

evaluate( test_loader, validation=False)
