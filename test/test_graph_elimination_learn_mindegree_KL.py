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
from gcn.models_gcn import GCN_Policy_SelectNode, GCN_Sparse_Policy_SelectNode, GAN
from data.graph import Graph
from utils import utils

# Training argument setting
parser = argparse.ArgumentParser()
parser.add_argument('--nocuda', action= 'store_true', default=False, help='Disable Cuda')
parser.add_argument('--novalidation', action= 'store_true', default=True, help='Disable validation')
parser.add_argument('--seed', type=int, default=50, help='Radom seed')
parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
parser.add_argument('--lr', type=float, default= 0.01, help='Learning rate')
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--dhidden', type=int, default=1, help='Dimension of hidden features')
parser.add_argument('--dinput', type=int, default=1, help='Dimension of input features')
parser.add_argument('--doutput', type=int, default=1, help='Dimension of output features')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout Rate')
parser.add_argument('--alpha', type=float, default=0.2, help='Aplha')
parser.add_argument('--nnode', type=int, default=100, help='Number of node per graph')
parser.add_argument('--ngraph', type=int, default=10, help='Number of graph per dataset')
parser.add_argument('--nnode_test', type=int, default=100, help='Number of node per graph for test')
parser.add_argument('--ngraph_test', type=int, default=1000, help='Number of graph for test dataset')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.cuda = not args.nocuda and torch.cuda.is_available()

if args.cuda:
   torch.cuda.manual_seed(args.seed)

def train(model, opt, train_loader, features, epoches=args.epochs, is_cuda = args.cuda):
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
        for x in X:
            for epoch in range(epoches):
                n = x.n
                x1 = Graph(x.M)
                total_loss_train_1graph = 0
                for i in range(n - 2):

                    features = np.ones([x1.n, args.dinput], dtype=np.float32)
                    m = torch.FloatTensor(x1.M)
                    m = utils.to_sparse(m)  # convert to coo sparse tensor
                    features = torch.FloatTensor(features)
                    degree = torch.FloatTensor(x1.min_degree_d()) # get the min-degree distribution of graph as label
                    if is_cuda:
                        m = m.cuda()
                        features = features.cuda()
                        degree = degree.cuda()

                    output = model(features, m)
                    output = output.view(-1)

                    loss_train = F.kl_div(output, degree) # get the negetive likelyhood
                    total_loss_train += loss_train.item()

                    opt.zero_grad()
                    loss_train.backward()
                    opt.step()

                    node_chosen, z = x1.min_degree(x1.M)  # get the node with minimum degree as label
                    edges_added = x1.eliminate_node(node_chosen, reduce=True)

                av_loss_train_1graph = total_loss_train_1graph / (n - 2)
                n_graphs_proceed += 1
                total_loss_train += av_loss_train_1graph


    av_loss_train = total_loss_train/n_graphs_proceed

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
            x = Graph(x.M)
            m = torch.FloatTensor(x.M)
            m = utils.to_sparse(m)  # convert to coo sparse tensor
            features = torch.FloatTensor(features)
            degree = torch.FloatTensor(x.min_degree_d())
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

def test(model, features, data_loader, is_cuda=args.cuda, validation = True):
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

    n_graphs_proceed = 0
    ratio = []
    for X in data_loader:
        _ratio = 0
        for x in X:
            n_e_mindegree = 0 # number of added edges
            # n_e_onestep = 0

            q_mindegree = np.zeros(x.n - 1, dtype=int) # index of ordering
            q2_mindegree = np.zeros(x.n - 1, dtype=int) # degree value of ordering
            q3_mindegree = np.zeros(x.n - 1, dtype=int) # number of edges added each step


            # q_onestep = np.zeros(x.n - 1, dtype=int)  # index of ordering
            # q2_onestep = np.zeros(x.n - 1, dtype=int) # number of edges to make neighbour-clique of ordering

            x1 = Graph(x.M)

            # x2 = Graph(x.M)

            for i in range(x.n-1):
                # choose the node with minimum degree
                node_chosen, d_min = x1.min_degree(x1.M)
                q_mindegree[i] = node_chosen
                q2_mindegree[i] = d_min

                # # choose the node with one step greedy
                # q_onestep[i], q2_onestep = x2.onestep_greedy()
                # eliminate the node chosen
                q3_mindegree[i] = x1.eliminate_node(q_mindegree[i], reduce=True)
                n_e_mindegree += q3_mindegree[i]

            n_e_baseline = n_e_mindegree
            # samles multi solutions by GCN and pick the lowest cost one
            samle_gcn = 10

            n_e_gcn_samples = np.zeros(samle_gcn)
            q_gcn_samples = np.zeros([samle_gcn, x.n - 1], dtype=int) # index of ordering given by GCN
            q3_gcn_samples = np.zeros([samle_gcn, x.n - 1], dtype=int) # number of edges added each step
            for j in range(samle_gcn):
                x3 = Graph(x.M)
                for i in range(x.n-1):

                    # choose the node with GCN
                    features = np.ones([x3.n,args.dinput], dtype=np.float32)
                    M_gcn = torch.FloatTensor(x3.M)
                    M_gcn = utils.to_sparse(M_gcn)  # convert to coo sparse tensor
                    features = torch.FloatTensor(features)
                    degree = torch.FloatTensor(x3.min_degree_d())
                    if is_cuda:
                        M_gcn = M_gcn.cuda()
                        features = features.cuda()
                        degree = degree.cuda()
                    output = model(features, M_gcn)
                    output = output.view(-1)
                    q_gcn_samples[j,i] = np.argmax(np.array(output.detach().numpy())) # choose the node given by GCN

                    q3_gcn_samples[j,i] = x3.eliminate_node(q_gcn_samples[j,i], reduce=True) # eliminate the node and return the number of edges added
                    n_e_gcn_samples[j] += q3_gcn_samples[j,i]
            k = np.argmin(n_e_gcn_samples)
            n_e_gcn = n_e_gcn_samples[k]
            q_gcn = q_gcn_samples[k,:]
            q3_gcn = q3_gcn_samples[k,:]

            _ratio = n_e_gcn/n_e_baseline
            ratio.append(_ratio)

                # n_e_onestep += x2.eliminate_node(q_onestep[i], reduce=True)

            # num_e = torch.IntTensor(x.num_e)

            # print('epoch {:04d}'.format(epoch),
            # 'min_degree number of edges {}'.format(n_e_mindegree))
            # print('epoch {:04d}'.format(epoch),
            # 'mindegree elimination ordering {}'.format(q_mindegree))
            # print('epoch {:04d}'.format(epoch),
            # 'mindegree elimination edge add {}'.format(q3_mindegree))
            #
            # print('epoch {:04d}'.format(epoch),
            # 'GCN number of edges {}'.format(n_e_gcn))
            # print('epoch {:04d}'.format(epoch),
            # 'GCN elimination ordering {}'.format(q_gcn))
            # print('epoch {:04d}'.format(epoch),
            # 'GCN elimination edge add {}'.format(q3_gcn))

            # print('epoch {:04d}'.format(epoch),
            # 'one_step_greedy number of edges {}'.format(n_e_onestep))
            # print('epoch {:04d}'.format(epoch),
            # 'one_step_greedy elimination ordering {}'.format(q_onestep))
            # print('epoch {:04d}'.format(epoch),
            # 'one_step_greedy {}'.format(q2_onestep))
        n_graphs_proceed += len(X)
    ratio = np.array(ratio).reshape(-1)
    total_ratio = np.sum(ratio)
    min_ratio = np.min(ratio)
    max_ratio = np.max(ratio)
    av_ratio = total_ratio/n_graphs_proceed

    return ratio, av_ratio, max_ratio, min_ratio


# load data and pre-process
train_set = ErgDataset(args.nnode, args.ngraph)
val_set = ErgDataset(args.nnode, args.ngraph)
test_set = ErgDataset(args.nnode_test, args.ngraph_test)

train_loader = DataLoader(train_set, batch_size=1, collate_fn=lambda x: x)
val_loader = DataLoader(val_set, batch_size=1, collate_fn=lambda x: x)
test_loader = DataLoader(test_set, batch_size=1, collate_fn=lambda x: x)

# build the GCN model
# model = GCN_Policy_SelectNode(nin=args.dinput,
#                               nhidden= args.dhidden,
#                               nout=args.doutput,
#                               dropout=args.dropout,
#                               ) # alpha=args.alpha
# build the GCN_Sparse model
# model = GCN_Sparse_Policy_SelectNode(nin=args.dinput,
#                               nhidden= args.dhidden,
#                               nout=args.doutput,
#                               dropout=args.dropout,
#                               ) # alpha=args.alpha
# build the GAN model
model = GAN(nin=args.dinput,
            nhidden= args.dhidden,
            nout=args.doutput,
            dropout=args.dropout,
            alpha=args.alpha
            ) # alpha=args.alpha


if args.cuda:
    model.cuda()

# optimizer setting
opt = optim.Adam(model.parameters(),
                 lr=args.lr,
                 weight_decay=args.wd)

# Train the model
features = np.ones([args.nnode,args.dinput], dtype=np.float32) # initialize the features for training set

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
# features = np.ones([args.nnode_test, args.dinput], dtype=np.float32) # initialize the features for test set
# av_loss_test = evaluate(model, test_loader, features, validation=False)

# print('Test result:',
#           'loss of test {:.4f}'.format(av_loss_test),
#           #'test accuracy {:.4f}'.format(av_acc_test)
#           )


ratio, av_ratio, max_ratio, min_ratio = test(model, features, test_loader, validation=False)
print('Test result:',
          'average ratio {:.4f}'.format(av_ratio),
          'max ratio {:.4f}'.format(max_ratio),
          'min ratio {:.4f}'.format(min_ratio)
          )

plt.hist(ratio, bins= 10)
plt.title('histogram : gnn2mindegree ratio kl')
plt.xlabel('loss ratio: model/heuristic')
plt.savefig('./results/histogram_gnn2mindegree_kl_gan.png')


