from data.ergDataset import ErgDataset
from torch.utils.data import DataLoader
import pickle as pkl
from utils.utils import save_dataset
import torch
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--nnode_lowerb', type=int, default=100, help='Number of node per graph')
parser.add_argument('--nnode_upperb', type=int, default=300, help='Number of graph per dataset')
parser.add_argument('--ngraph', type=int, default=20, help='Number of graph per dataset')
parser.add_argument('--p_lowerb', type=int, default=0.1, help='probiblity of edges')
parser.add_argument('--p_upperb', type=int, default=0.3, help='probiblity of edges')

args = parser.parse_args()

dataset = ErgDataset

train_dataset = dataset(n_nodes_lowerb=args.nnode_lowerb, n_nodes_upperb=args.nnode_upperb, p_lowerb=args.p_lowerb,
                        p_upperb=args.p_upperb, n_graphs=args.ngraph, random_seed= 32)
val_dataset =dataset(n_nodes_lowerb=args.nnode_lowerb, n_nodes_upperb=args.nnode_upperb, p_lowerb=args.p_lowerb,
                     p_upperb=args.p_upperb, n_graphs=args.ngraph, random_seed= 33)
test_dataset = dataset(n_nodes_lowerb=args.nnode_lowerb, n_nodes_upperb=args.nnode_upperb, p_lowerb=args.p_lowerb,
                      p_upperb=args.p_upperb, n_graphs=args.ngraph, random_seed=34)

# (Each has 200 graphs) ER small: 100-300; ER mid:300:500; ER large:500-700
save_dataset('./data/ERGcollection/erg_small_20graphs.pkl', train_dataset, val_dataset, test_dataset)

# def save(filename, train, val, test):
#     with open(filename, "wb") as f:
#         pkl.dump([train,val, test], f)
#
# def open_file(filename):
#     with open(filename, "rb") as f:
#         train, val, test  = pkl.load(f)
#     return train, val, test

# train, val, test = open_file('erg_small.pkl')
# train_loader = DataLoader(train, shuffle=True, batch_size=1, collate_fn=lambda x: x)
# val_loader = DataLoader(val, shuffle=True, batch_size=1, collate_fn=lambda x: x)
# test_loader = DataLoader(test, shuffle=True, batch_size=1, collate_fn=lambda x: x)
#
# for X in train_loader:
#     _ratio = 0
#     for x in X:
#         print(x.M)
#
# for X in val_loader:
#     _ratio = 0
#
#     for x in X:
#         print(x.M)
#
# for X in test_loader:
#     _ratio = 0
#
#     for x in X:
#         print(x.M)
#





