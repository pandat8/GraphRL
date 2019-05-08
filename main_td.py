from torch.utils.data import Dataset, DataLoader
from data.graphDataset import GraphDataset
from data.SSMCDataset import SSMCDataset
from data.UFSMDataset import UFSMDataset

from rl.model_a2c import Model_A2C, Model_A2C_Sparse
from rl.train_a2c_td import TrainModel_TD
from rl.train_a2c_mc import TrainModel_MC
from gcn.models_gcn import GCN_Policy_SelectNode, GCN_Sparse_Policy_SelectNode, GAN, GAN_Value
from gcn.models_gcn import GCN_Value, GCN_Sparse_Value

import sys
import time
import argparse
import torch
import numpy as np


# Training argument setting
parser = argparse.ArgumentParser()
parser.add_argument('--nocuda', action= 'store_true', default=False, help='Disable Cuda')
parser.add_argument('--novalidation', action= 'store_true', default=False, help='Disable validation')
parser.add_argument('--seed', type=int, default=50, help='Radom seed')
parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
parser.add_argument('--lr_actor', type=float, default= 0.01, help='Learning rate of actor')
parser.add_argument('--lr_critic', type=float, default= 0.01, help='Learning rate of critic')
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--dhidden', type=int, default=1, help='Dimension of hidden features')
parser.add_argument('--dinput', type=int, default=1, help='Dimension of input features')
parser.add_argument('--doutput', type=int, default=1, help='Dimension of output features')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout Rate')
parser.add_argument('--alpha', type=float, default=0.2, help='Aplha')
parser.add_argument('--nnode', type=int, default=200, help='Number of node per graph')
parser.add_argument('--ngraph', type=int, default=100, help='Number of graph per dataset')
parser.add_argument('--nnode_test', type=int, default=1000, help='Number of node per graph for test')
parser.add_argument('--ngraph_test', type=int, default=1, help='Number of graph for test dataset')
parser.add_argument('--use_critic', type=bool, default=True, help='Number of graph for test dataset')
parser.add_argument('--use_gan', type=bool, default=True, help='GNN model type, use GCN if False')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.cuda = not args.nocuda and torch.cuda.is_available()

if args.cuda:
   torch.cuda.manual_seed(args.seed)

# load data and pre-process

train_dataset = GraphDataset(args.nnode, args.ngraph)
val_dataset = GraphDataset(args.nnode, args.ngraph)
# test_dataset = GraphDataset(args.nnode_test, args.ngraph_test)
# train_dataset = UFSMDataset()
#test_dataset = SSMCDataset()

# test_loader = DataLoader(test_set, batch_size=1, collate_fn=lambda x: x)

# build the GCN model
if args.use_gan:
    # build the GAN model
    actor = GAN(nin=args.dinput,
                nhidden=args.dhidden,
                nout=args.doutput,
                dropout=args.dropout,
                alpha=args.alpha
                )  # alpha=args.alpha
else:
    actor = GCN_Sparse_Policy_SelectNode(nin=args.dinput,
                              nhidden= args.dhidden,
                              nout=args.doutput,
                              dropout=args.dropout,
                              ) # alpha=args.alpha



critic = None
if args.use_critic:

    if args.use_gan:
        critic = GAN_Value(nin=args.dinput,
                nhidden=args.dhidden,
                nout=args.doutput,
                dropout=args.dropout,
                alpha=args.alpha
                )  # alpha=args.alpha
    else:
        critic = GCN_Sparse_Value(nin=args.dinput,
                                  nhidden=args.dhidden,
                                  nout=args.doutput,
                                  dropout=args.dropout,
                                  ) # alpha=args.alpha

model_a2c = Model_A2C_Sparse(actor=actor,
                             use_critic= args.use_critic, #
                             use_cuda= args.cuda,
                             critic= critic)
if args.cuda:
   model_a2c.cuda()

# train the model
train_a2c = TrainModel_TD(model_a2c,
                          train_dataset,
                          val_dataset,
                          use_cuda=args.cuda)
print('Training started')
time_start = time.time()
train_a2c.train_and_validate(n_epochs=args.epochs,
                                use_critic= args.use_critic,
                                    lr_actor=args.lr_actor,
                                    lr_critic=args.lr_critic)
print('Training finished')
print('Training time: {:.4f}'.format(time.time()-time_start))





