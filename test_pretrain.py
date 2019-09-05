import numpy as np
import argparse
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from torch.utils.data import Dataset, DataLoader
from data.ergDataset import ErgDataset
from utils.utils import open_dataset
from data.SSMCDataset import SSMCDataset
from data.UFSMDataset import UFSMDataset
from gcn.models_gcn import GCN_Policy_SelectNode, GCN_Sparse_Policy_SelectNode, GCN_Sparse_Memory_Policy_SelectNode
from supervised.train_supervised_learning import Train_SupervisedLearning
from data.graph import Graph

# Training argument setting
parser = argparse.ArgumentParser()
parser.add_argument('--nocuda', action= 'store_true', default=False, help='Disable Cuda')
parser.add_argument('--novalidation', action= 'store_true', default=True, help='Disable validation')
parser.add_argument('--seed', type=int, default=50, help='Radom seed')
parser.add_argument('--epochs', type=int, default=41, help='Training epochs')
parser.add_argument('--lr', type=float, default= 0.0001, help='Learning rate')
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--dhidden', type=int, default=1, help='Dimension of hidden features')
parser.add_argument('--dinput', type=int, default=1, help='Dimension of input features')
parser.add_argument('--doutput', type=int, default=1, help='Dimension of output features')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout Rate')
parser.add_argument('--alpha', type=float, default=0.2, help='Aplha')
parser.add_argument('--nnode', type=int, default=200, help='Number of node per graph')
parser.add_argument('--ngraph', type=int, default=200, help='Number of graph per dataset')
parser.add_argument('--p', type=int, default=0.1, help='probiblity of edges')
parser.add_argument('--nnode_test', type=int, default=300, help='Number of node per graph for test')
parser.add_argument('--ngraph_test', type=int, default=100, help='Number of graph for test dataset')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.cuda = not args.nocuda and torch.cuda.is_available()

if args.cuda:
   torch.cuda.manual_seed(args.seed)

# def train(model, opt, train_loader, features, epoches=args.epochs,  is_cuda = args.cuda):
#     """
#     Training function
#     :param model: neural network model
#     :param opt: optimizaiton function
#     :param train_loader: training dataset loader
#     :param features: features vector
#     :param is_cuda:
#     :return: averaged loss per graph
#     """
#
#     model.train()
#
#     total_acc_train = 0
#     n_graphs_proceed = 0
#     # n_iteration = 0
#     total_loss_train = 0
#     for X in train_loader:
#         for x in X:
#             for epoch in range(epoches):
#
#                 n = x.n
#                 x1 = Graph(x.M)
#                 total_loss_train_1graph = 0
#                 # edges_total = 0
#                 for i in range(n - 2):
#
#                     features = np.ones([x1.n, args.dinput], dtype=np.float32)
#                     m = torch.FloatTensor(x1.M)
#                     features = torch.FloatTensor(features)
#                     node_chosen, z = x1.min_degree(x1.M) # get the node with minimum degree as label
#                     node_chosen = torch.from_numpy(np.array(node_chosen)) # one-hot coding
#                     node_chosen = node_chosen.reshape(1)
#                     if is_cuda:
#                         m = m.cuda()
#                         features = features.cuda()
#                         node_chosen = node_chosen.cuda()
#
#                     output = model(features, m)
#
#                     output = output.t()
#
#                     # m = Categorical(output)
#                     # node_selected = m.sample()
#                     # node_selected = torch.LongTensor([[node_selected]])
#                     # m.probs.zero_()
#                     # m.probs.scatter_(1, node_selected, 1)
#
#                     loss_train = F.cross_entropy(output, node_chosen) # get the negetive likelyhood
#                     total_loss_train_1graph += loss_train.item()
#
#                     opt.zero_grad()
#                     loss_train.backward()
#                     opt.step()
#
#                     action_gcn = np.argmax(np.array(output.detach().cpu().numpy()))  # choose the node given by GCN
#                     edges_added = x1.eliminate_node(action_gcn, reduce=True)
#                     # edges_total +=edges_added
#
#                 av_loss_train_1graph = total_loss_train_1graph / (n-2)
#                 n_graphs_proceed += 1
#                 total_loss_train += av_loss_train_1graph
#
#     av_loss_train = total_loss_train/(n_graphs_proceed)
#
#     return av_loss_train

# def evaluate(model, data_loader, features, is_cuda=args.cuda, validation = False):
#     """
#     Evaluation function
#     :param model: network model
#     :param data_loader: dataset loader depending on validation or test
#     :param features: initial feature vector of graph
#     :param is_cuda:
#     :param validation: True if validation(by default), False if test
#     :return: averaged loss per graph
#     """
#     model.eval()
#     total_acc_train = 0
#     n_graphs_proceed = 0
#     total_loss_train = 0
#     total_acc = 0
#     total_loss = 0
#     n_graphs_proceed = 0
#     for X in train_loader:
#         for x in X:
#             x = Graph(x.M)
#             total_loss_train_1graph = 0
#             n = x.n
#             for i in range(n - 2):
#
#                 features = np.ones([x.n, args.dinput], dtype=np.float32)
#                 m = torch.FloatTensor(x.M)
#                 features = torch.FloatTensor(features)
#                 node_chosen, z = x.min_degree(x.M)  # get the node with minimum degree as label
#                 node_chosen = torch.from_numpy(np.array(node_chosen))  # one-hot coding
#                 node_chosen = node_chosen.reshape(1)
#                 if is_cuda:
#                     m = m.cuda()
#                     features = features.cuda()
#                     node_chosen = node_chosen.cuda()
#
#                 output = model(features, m)
#                 output = output.t()
#                 print('epoch {:04d}'.format(epoch),
#                       'output {}'.format(output.exp()))
#                 print('epoch {:04d}'.format(epoch),
#                       'degree {}'.format(node_chosen))
#                 loss_train = F.cross_entropy(output, node_chosen)  # get the negetive likelyhood
#                 total_loss_train_1graph += loss_train.item()
#                 if not validation:
#                     plot_dis(output.exp().detach().numpy(), node_chosen.detach().cpu().numpy())
#
#                 action_gcn = np.argmax(np.array(output.detach().numpy()))  # choose the node given by GCN
#                 edges_added = x.eliminate_node(action_gcn, reduce=True)
#
#             av_loss_train_1graph = total_loss_train_1graph / (n - 2)
#             n_graphs_proceed += 1
#             total_loss_train += av_loss_train_1graph
#
#     av_loss_train = total_loss_train / (n_graphs_proceed)
#
#     return av_loss_train
#
#     # for X in data_loader:
#     #     for x in X:
#     #         m = torch.FloatTensor(x.M)
#     #         features = torch.FloatTensor(features)
#     #         node_chosen, z = x.onestep_greedy()  # get the min-degree distribution of graph as label
#     #         node_chosen = torch.from_numpy(np.array(node_chosen))  # one-hot coding
#     #         node_chosen = node_chosen.reshape(1)
#     #         if is_cuda:
#     #             m = m.cuda()
#     #             features = features.cuda()
#     #             node_chosen = node_chosen.cuda()
#     #         output = model(features, m)
#     #         output = output.t()
#     #         print('epoch {:04d}'.format(epoch),
#     #               'output {}'.format(output.exp()))
#     #         print('epoch {:04d}'.format(epoch),
#     #               'degree {}'.format(node_chosen))
#     #         loss = F.cross_entropy(output, node_chosen)  # get the negetive likelyhood
#     #         total_loss += loss.item()
#     #
#     #         if not validation:
#     #             plot_dis(output.exp().detach().numpy(), node_chosen.detach().numpy())
#     #         # acc = accuracy(output, degree)
#     #         # total_acc += acc
#     #     n_graphs_proceed += len(X)
#     #
#     # av_loss = total_loss / n_graphs_proceed
#     # # av_acc = total_acc/n_graphs_proceed
#     # return av_loss


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

# def test(model, features, data_loader, is_cuda=args.cuda, validation = True):
#     """
#     Evaluation function
#     :param model: network model
#     :param data_loader: dataset loader depending on validation or test
#     :param features: initial feature vector of graph
#     :param is_cuda:
#     :param validation: True if validation(by default), False if test
#     :return: averaged loss per graph
#     """
#     model.eval()
#
#     n_graphs_proceed = 0
#     ratio = []
#     ratio_g2r = []
#     for X in data_loader:
#         _ratio = 0
#         for x in X:
#             n_e_mindegree = 0 # number of added edges
#             n_e_random = 0
#             # n_e_onestep = 0
#
#             q_mindegree = np.zeros(x.n - 1, dtype=int) # index of ordering
#             q2_mindegree = np.zeros(x.n - 1, dtype=int) # degree value of ordering
#             q3_mindegree = np.zeros(x.n - 1, dtype=int) # number of edges added each step
#
#             q_random = np.zeros(x.n - 1, dtype=int) # index of ordering
#             q3_random = np.zeros(x.n - 1, dtype=int) # number of edges added each step
#
#
#             # q_onestep = np.zeros(x.n - 1, dtype=int)  # index of ordering
#             # q2_onestep = np.zeros(x.n - 1, dtype=int) # number of edges to make neighbour-clique of ordering
#
#             x1 = Graph(x.M)
#             x4 = Graph(x.M)
#
#             # x2 = Graph(x.M)
#
#             for i in range(x.n-2):
#                 # choose the node with minimum degree
#                 node_chosen, d_min = x1.min_degree(x1.M)
#                 q_mindegree[i] = node_chosen
#                 q2_mindegree[i] = d_min
#
#                 # random eliminate a node
#                 q_random[i] = np.random.randint(low=0, high=x4.n)
#                 q3_random[i] = x4.eliminate_node(q_random[i], reduce=True)
#                 n_e_random += q3_random[i]
#
#                 # # choose the node with one step greedy
#                 # q_onestep[i], q2_onestep = x2.onestep_greedy()
#                 # eliminate the node chosen
#                 q3_mindegree[i] = x1.eliminate_node(q_mindegree[i], reduce=True)
#                 n_e_mindegree += q3_mindegree[i]
#
#             n_e_baseline = n_e_mindegree
#             # samles multi solutions by GCN and pick the lowest cost one
#             samle_gcn = 10
#
#             edges_total_samples = np.zeros(samle_gcn)
#             q_gcn_samples = np.zeros([samle_gcn, x.n - 1], dtype=int) # index of ordering given by GCN
#             edges_added = np.zeros([samle_gcn, x.n - 1], dtype=int) # number of edges added each step
#             for j in range(samle_gcn):
#                 x3 = Graph(x.M)
#                 for i in range(x.n-2):
#
#                     # choose the node with GCN
#                     features = np.ones([x3.n,args.dinput], dtype=np.float32)
#                     M_gcn = torch.FloatTensor(x3.M)
#                     features = torch.FloatTensor(features)
#                     if is_cuda:
#                         M_gcn = M_gcn.cuda()
#                         features = features.cuda()
#                     output = model(features, M_gcn)
#                     output = output.view(-1)
#                     m = Categorical(output)
#                     node_selected = m.sample()
#                     q_gcn_samples[j,i] = node_selected # choose the node given by GCN
#
#                     edges_added[j,i] = x3.eliminate_node(q_gcn_samples[j,i], reduce=True) # eliminate the node and return the number of edges added
#                     edges_total_samples[j] += edges_added[j,i]
#             k = np.argmin(edges_total_samples)
#             n_e_gcn = edges_total_samples[k]
#
#             q_gcn = q_gcn_samples[k,:]
#             q3_gcn = edges_added[k,:]
#
#             _ratio = n_e_gcn/n_e_baseline
#             _ratio_g2r = n_e_gcn/n_e_random
#             ratio.append(_ratio)
#             ratio_g2r.append(_ratio_g2r)
#             #print('GCN number of edges {}'.format(np.sum(q3_gcn)))
#             #print('Ran number of edges {}'.format(n_e_random))
#
#                 # n_e_onestep += x2.eliminate_node(q_onestep[i], reduce=True)
#
#             # num_e = torch.IntTensor(x.num_e)
#
#             # print('epoch {:04d}'.format(epoch),
#             # 'min_degree number of edges {}'.format(n_e_mindegree))
#             # print('epoch {:04d}'.format(epoch),
#             # 'mindegree elimination ordering {}'.format(q_mindegree))
#             # print('epoch {:04d}'.format(epoch),
#             # 'mindegree elimination edge add {}'.format(q3_mindegree))
#             #
#             # print('epoch {:04d}'.format(epoch),
#             # 'GCN number of edges {}'.format(n_e_gcn))
#             # print('epoch {:04d}'.format(epoch),
#             # 'GCN elimination ordering {}'.format(q_gcn))
#             # print('epoch {:04d}'.format(epoch),
#             # 'GCN elimination edge add {}'.format(q3_gcn))
#
#             # print('epoch {:04d}'.format(epoch),
#             # 'one_step_greedy number of edges {}'.format(n_e_onestep))
#             # print('epoch {:04d}'.format(epoch),
#             # 'one_step_greedy elimination ordering {}'.format(q_onestep))
#             # print('epoch {:04d}'.format(epoch),
#             # 'one_step_greedy {}'.format(q2_onestep))
#         n_graphs_proceed += len(X)
#     ratio = np.array(ratio).reshape(-1)
#     ratio_g2r = np.array(ratio_g2r).reshape(-1)
#
#     total_ratio = np.sum(ratio)
#     total_ratio_g2r = np.sum(ratio_g2r)
#
#     min_ratio = np.min(ratio)
#     max_ratio = np.max(ratio)
#     av_ratio = total_ratio/n_graphs_proceed
#
#     min_ratio_g2r = np.min(ratio_g2r)
#     max_ratio_g2r = np.max(ratio_g2r)
#     av_ratio_g2r = total_ratio_g2r / n_graphs_proceed
#
#     return ratio, av_ratio, max_ratio, min_ratio, ratio_g2r, av_ratio_g2r, max_ratio_g2r, min_ratio_g2r


#  load data and pre-process
# train_dataset = GraphDataset(args.nnode, args.ngraph)

dataset = ErgDataset
if dataset.__name__ == 'UFSMDataset':
    train_dataset = dataset(start=18, end=19)
    val_dataset = dataset(start=19, end=20)
    test_dataset = dataset(start=24, end=26)
elif dataset.__name__ == 'ErgDataset':
    train_dataset, val_dataset, test_dataset = open_dataset('./data/ERGcollection/erg_small.pkl')

# build the GCN model
model = GCN_Sparse_Policy_SelectNode(nin=args.dinput,
                              nhidden= args.dhidden,
                              nout=args.doutput,
                              dropout=args.dropout,
                              ) # alpha=args.alpha
if args.cuda:
    model.cuda()

heuristic = 'min_degree' # 'one_step_greedy' 'min_degree'
policy_sl = Train_SupervisedLearning(model=model, heuristic=heuristic, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, use_cuda = args.cuda)


# Train the model
print('Supervised Training started')
print('heuristic: '+heuristic,
      'learning rate: {}'.format(args.lr),
      'epochs: {}'.format(args.epochs),
      'DataSet: '+train_dataset.__class__.__name__+'\n')

time_start = time.time()
t = time.time()
av_loss_train,t_model_opt, t_IO , t_heu, t_eli, t_spa, t_all = policy_sl.train(epochs=args.epochs, lr=args.lr)
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
print('Training finished')
#print('Training time: {:.4f}'.format(time_end-time_start))
print('Training time: {:.4f}'.format(t_all))
print('Elimination time: {:.4f}'.format(t_eli))
print('Heuristic'+heuristic+' time: {:.4f}'.format(t_heu))
print('Dense 2 Sparce time: {:.4f}'.format(t_spa))
print('IO to cuda time: {:.4f}'.format(t_IO))
print('Model and Opt time: {:.4f}'.format(t_model_opt))



if args.cuda:
    torch.save(model.state_dict(), './results/models/gcn_policy_'+heuristic+'_pre_'+dataset.__name__+str(args.nnode)+'dense_'+str(args.p)+'_epochs'+str(args.epochs)+'_cuda.pth')
    # torch.save(model.state_dict(),
    #            './results/models/gcn_policy_' + heuristic + '_pre_' + dataset.__name__ + '_epochs' + str(
    #                args.epochs) + '_cuda.pth')
else:
    torch.save(model.state_dict(), './results/models/gcn_policy_min_degree_pre_erg100.pth')


# # Test the model
# model_test = GCN_Sparse_Policy_SelectNode(nin=args.dinput,
#                               nhidden= args.dhidden,
#                               nout=args.doutput,
#                               dropout=args.dropout,
#                               ) # alpha=args.alpha
#
# if args.cuda:
#     model_test.load_state_dict(torch.load('./results/models/gcn_policy_' + heuristic + '_pre_' + dataset.__name__ +str(args.nnode)+ 'dense_' + str(
#                    args.p) + '_epochs' + str(args.epochs) + '_cuda.pth'))
#
#     # model_test.load_state_dict(torch.load('./results/models/gcn_policy_'+heuristic+'_pre_'+dataset.__name__+'_epochs'+str(args.epochs)+'_cuda.pth'))
#     model_test.cuda()
# else:
#     model_test.load_state_dict(torch.load('./results/models/gcn_policy_min_degree_pre_erg100.pth'))
# epoch = 0
# # features = np.ones([args.nnode_test, args.dinput], dtype=np.float32) # initialize the features for test set
# # av_loss_test = evaluate(model, test_loader, features, validation=False)
#
# # print('Test result:',
# #           'loss of test {:.4f}'.format(av_loss_test),
# #           #'test accuracy {:.4f}'.format(av_acc_test)
# #           )
#
# # test trained policy model
# policy_sl.model = model_test
#
# print('test stated')
# time_start = time.time()
# ratio, av_ratio, max_ratio, min_ratio, ratio_g2r, av_ratio_g2r, max_ratio_g2r, min_ratio_g2r = policy_sl.test()
# time_end = time.time()
# print('test finished')
# print('Test time: {:.4f}'.format(time_end-time_start))
# if args.cuda:
#     text_file = open("test/results/pretrain_"+heuristic+"_gcn_"+dataset.__name__+"_cuda.txt", "w")
# else:
#     text_file = open("test/results/pretrain_min_degree_gcn_memory_ERG100.txt", "w")
# text_file.write('\n Test result: test_graph_elimination_learn_'+heuristic+'\n')
# text_file.write('DataSet: '+dataset.__name__+'\n')
# text_file.write('average ratio gcn2heuristic {:.4f} \n'.format(av_ratio))
# text_file.write('max ratio gcn2heuristic {:.4f}\n'.format(max_ratio))
# text_file.write('min ratio {:.4f}\n'.format(min_ratio))
# text_file.write('average ratio gcn2random {:.4f}\n'.format(av_ratio_g2r))
# text_file.write('max ratio gcn2random {:.4f}\n'.format(max_ratio_g2r))
# text_file.write('min ratio gcn2random {:.4f}\n'.format(min_ratio_g2r))
# text_file.close()
#
# if args.cuda:
#     plt.switch_backend('agg')
#     plt.hist(ratio, bins=32)
#     plt.title('histogram: gcn2'+heuristic+' ratio of '+ dataset.__name__)
#     plt.savefig('./test/results/histogram_gnn2'+heuristic+'_gcn_logsoftmax_'+dataset.__name__+str(args.nnode_test)+'_dense_' + str(args.p)+'__cuda.png')
#     plt.clf()
#     #
#     plt.hist(ratio_g2r, bins=32)
#     plt.title('gcn2random ratio Erdos-Renyi graph')
#     plt.savefig('./test/results/histogram_gnn2random_gcn_logsoftmax_'+heuristic+'_'+dataset.__name__+str(args.nnode_test)+'_dense_' + str(args.p)+'__cuda.png')
#     plt.clf()
# else:
#     plt.hist(ratio, bins= 32)
#     plt.title('histogram: gcn2mindegree ratio CrossEntropy of Erdos-Renyi graph')
#     plt.savefig('./test/results/histogram_gnn2mindegree_gcn_memory_erg100.png')
#     plt.clf()
#     #
#     plt.hist(ratio_g2r, bins= 32)
#     plt.title('gcn2random_ratio_CrossEntropy Erdos-Renyi graph')
#     plt.savefig('./test/results/histogram_gnn2random_gcn_memory_erg100.png')
#     plt.clf()
# #
# # plt.show()


