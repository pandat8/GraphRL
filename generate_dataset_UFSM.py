
from data.UFSMDataset import UFSMDataset
from torch.utils.data import DataLoader
import pickle as pkl
from utils.utils import save_dataset, dataset_split
import torch
import argparse



dataset = UFSMDataset
dataset_sub = 'ss_large'
dataset_dir = './data/UFSM/'+dataset_sub+'/'+dataset_sub+'_set/'
ufsm_dataset =dataset(input_dir=dataset_dir)
train_dataset, val_dataset, test_dataset  = dataset_split(dataset= ufsm_dataset, train_frac=0.4)

save_dataset('./data/UFSM/'+dataset_sub+'/'+dataset_sub+'_split.pkl', train_dataset, val_dataset, test_dataset)
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





