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
parser.add_argument('--ngraph', type=int, default=20, help='Number of graph per dataset')
parser.add_argument('--nnode_test', type=int, default=20, help='Number of node per graph for test')
parser.add_argument('--ngraph_test', type=int, default=2, help='Number of graph for test dataset')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.cuda = not args.nocuda and torch.cuda.is_available()

if args.cuda:
   torch.cuda.manual_seed(args.seed)

def train(model, opt, train_loader, features,  is_cuda = args.cuda):
    """
    Training function
    :param model: neural network model
    :param opt: optimizaiton function
    :param train_loader: training dataset loader
    :param features: features vector
    :param is_cuda:
    :return: averaged loss per graph
    """

    model.train()

    total_loss_train = 0
    total_acc_train = 0
    n_graphs_proceed = 0
    for X in train_loader:
        opt.zero_grad()
        for x in X:
            m = torch.FloatTensor(x.M)
            features = torch.FloatTensor(features)
            degree = torch.FloatTensor(x.degree_d) # get the degree distribution of graph as label

            if is_cuda:
                m = m.cuda()
                features = features.cuda()
                degree = degree.cuda()

            output = model(features, m)
            output = output.view(-1)
            # print('epoch {:04d}'.format(epoch),
            # 'output {}'.format(output.exp()))
            # print('epoch {:04d}'.format(epoch),
            #     'degree {}'.format(degree))
            loss_train = F.kl_div(output, degree) # get the negetive likelyhood
            total_loss_train += loss_train.item()

            #acc_train = accuracy(output, degree) # get the accuracy rate of the output
            #total_acc_train += acc_train

            loss_train.backward()
        opt.step()
        n_graphs_proceed += len(X)

    av_loss_train = total_acc_train/n_graphs_proceed
    #av_acc_train = total_acc_train/n_graphs_proceed

    return av_loss_train

def evaluate(model, data_loader, features, is_cuda=args.cuda, validation = True):
    """
    Evaluation function
    :param model: network model
    :param data_loader: dataset loader depending on validation or test
    :param features: initial feature vector of graph
    :param is_cuda:
    :param validation: True if validation(by default), False if test
    :return: averaged loss per graph
    """
    model.eval()
    total_acc = 0
    total_loss = 0
    n_graphs_proceed = 0
    for X in data_loader:
        for x in X:
            m = torch.FloatTensor(x.M)
            features = torch.FloatTensor(features)
            degree = torch.FloatTensor(x.degree_d)
            if is_cuda:
                m = m.cuda()
                features = features.cuda()
                degree = degree.cuda()
            output = model(features, m)
            output = output.view(-1)
            print('epoch {:04d}'.format(epoch),
            'output {}'.format(output.exp()))
            print('epoch {:04d}'.format(epoch),
                'degree {}'.format(degree))
            loss = F.kl_div(output, degree) # get the negetive likelyhood
            total_loss += loss.item()
            if not validation:
                plot_dis(output.exp().detach().numpy(), degree.detach().numpy())
            #acc = accuracy(output, degree)
            #total_acc += acc
        n_graphs_proceed += len(X)

    av_loss = total_loss/n_graphs_proceed
    #av_acc = total_acc/n_graphs_proceed
    return av_loss

def accuracy(output, labels):
    """
    calculate the accuracy rate of the prediction output
    """
    preds = output.max(1)[1].type_as(labels) # get the prediction class
    correct = preds.eq(labels).double()
    correct_number = correct.sum()
    return correct_number/len(labels)

def plot_dis(output, labels):
    """
    plot the distribution of output and labels
    """
    plt.figure(1)
    plt.plot(output,'r',label='output-100epoch')
    plt.plot(labels,'b--',label='label distri')
    plt.legend()
    plt.show()


# load data and pre-process
train_set = ErgDataset(args.nnode, args.ngraph)
val_set = ErgDataset(args.nnode, args.ngraph)
test_set = ErgDataset(args.nnode_test, args.ngraph_test)

train_loader = DataLoader(train_set, batch_size=1, collate_fn=lambda x: x)
val_loader = DataLoader(val_set, batch_size=1, collate_fn=lambda x: x)
test_loader = DataLoader(test_set, batch_size=1, collate_fn=lambda x: x)

nin = 1  # length of input feature vector per node
nout = 1  # length of output feature vector per node

# build the GCN model
model = GCN_Policy_SelectNode(nin=nin,
                              nhidden= args.nhidden,
                              nout=nout,
                              dropout=args.dropout,
                              ) # alpha=args.alpha
if args.cuda:
    model.cuda()

# optimizer setting
opt = optim.Adam(model.parameters(),
                 lr=args.lr,
                 weight_decay=args.wd)

# Train the model
features = np.ones([args.nnode,nin], dtype=np.float32) # initialize the features

time_start = time.time()
for epoch in range(args.epochs):
    t = time.time()
    av_loss_train = train(model, opt, train_loader, features)
    if not args.novalidation:
        av_loss_val = evaluate(model, val_loader, features)
    # print('epoch {:04d}'.format(epoch),
    #       'loss of train {:4f}'.format(av_loss_train),
    #       #'train accuracy {:.4f}'.format(av_acc_train),
    #       'loss of val {:.4f}'.format(av_loss_val),
    #       #'val accuracy {:.4f}'.format(av_acc_val),
    #       'time {:.4f}'.format(time.time()-t)
    #     )

print('Training finished')
print('Training time: {:.4f}'.format(time.time()-time_start))

# Test the model
epoch = 0
features = np.ones([args.nnode_test, nin], dtype=np.float32) # initialize the features
av_loss_test = evaluate(model, test_loader, features, validation=False)

print('Test result:',
          'loss of test {:.4f}'.format(av_loss_test),
          #'test accuracy {:.4f}'.format(av_acc_test)
          )




