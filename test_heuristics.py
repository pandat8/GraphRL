import numpy as np
import argparse
import torch

import time

from data.ergDataset import ErgDataset

from data.test_performance import Test
from utils.utils import open_dataset

# Training argument setting
parser = argparse.ArgumentParser()
parser.add_argument('--nocuda', action= 'store_true', default=False, help='Disable Cuda')
parser.add_argument('--novalidation', action= 'store_true', default=True, help='Disable validation')
parser.add_argument('--seed', type=int, default=50, help='Radom seed')
parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
parser.add_argument('--lr', type=float, default= 0.001, help='Learning rate')
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--dhidden', type=int, default=1, help='Dimension of hidden features')
parser.add_argument('--dinput', type=int, default=1, help='Dimension of input features')
parser.add_argument('--doutput', type=int, default=1, help='Dimension of output features')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout Rate')
parser.add_argument('--alpha', type=float, default=0.2, help='Aplha')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.cuda = not args.nocuda and torch.cuda.is_available()

if args.cuda:
   torch.cuda.manual_seed(args.seed)

print('loading dataset')
dataset = ErgDataset
if dataset.__name__ == 'UFSMDataset':
    train_dataset = dataset(start=18, end=21)
    val_dataset = dataset(start=21, end=24)
    test_dataset = dataset(start=18, end=26)
elif dataset.__name__ == 'ErgDataset':
    train_dataset, val_dataset, test_dataset = open_dataset('./data/ERGcollection/erg_small.pkl')

print('Training Dataset with size {}'.format(train_dataset.__len__()))
print('Validation Dataset with size {}'.format(val_dataset.__len__()))
print('Test Dataset loaded with size {}'.format(test_dataset.__len__()))



def testing_dataset(dataset):

    test = Test(test_dataset= dataset)
    print('Testing is starting')
    time_start = time.time()
    t = time.time()
    test.test_heuristics()
        # if not args.novalidation:
        #     av_loss_val = evaluate(model, val_loader, features)
        # print('epoch {:04d}'.format(epoch),
        #       'loss of train {:4f}'.format(av_loss_train),
        #       #'train accuracy {:.4f}'.format(av_acc_train),
        #       'loss of val {:.4f}'.format(av_loss_val),
        #       #'val accuracy {:.4f}'.format(av_acc_val),
        #       'time {:.4f}'.format(time.time()-t)
        #     )
    time_end = time.time()
    print('Testing is finished')
    #print('Training time: {:.4f}'.format(time_end-time_start))
    print('Test time: {:.4f}'.format(time_end - t))

print('Training Set is starting')
testing_dataset(train_dataset)

print('Validation Set is starting')
testing_dataset(val_dataset)

print('Testing Set is starting')
testing_dataset(test_dataset)







