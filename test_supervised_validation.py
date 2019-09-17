import numpy as np
import argparse
import torch

import matplotlib.pyplot as plt
import inspect, re
import pickle as pkl

from data.ergDataset import ErgDataset
from utils.utils import open_dataset, varname

from data.UFSMDataset import UFSMDataset
from gcn.models_gcn import GCN_Policy_SelectNode, GCN_Sparse_Policy_SelectNode, GCN_Sparse_Memory_Policy_SelectNode
from supervised.train_supervised_learning import Train_SupervisedLearning
from data.graph import Graph

# Training argument setting
parser = argparse.ArgumentParser()
parser.add_argument('--nocuda', action= 'store_true', default=False, help='Disable Cuda')
parser.add_argument('--novalidation', action= 'store_true', default=True, help='Disable validation')
parser.add_argument('--seed', type=int, default=50, help='Radom seed')
parser.add_argument('--epochs', type=int, default=21, help='Training epochs')
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

def plot_performance_supervised(dataset_type ='val', steps=None, t_plot=None, val_ave_gcn_np=None, val_ave_mind_np=None, val_ave_rand_np=None):

    plt.clf()
    plt.plot(t_plot, val_ave_gcn_np, t_plot, val_ave_mind_np, t_plot, val_ave_rand_np)
    plt.legend(('GNN', heuristic, 'random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
               loc='upper right')  # 'GNN-initial', 'GNN-RL', 'min-degree'
    plt.title(
        'Performance on ' + dataset_type + ' ' + val_dataset.__class__.__name__ + ' (average number of filled edges)')
    plt.ylabel('number of fill-in')
    # plt.draw()
    plt.savefig(
        './results/supervised/'+heuristic+'/lr' + str(args.lr) + '/final_' + dataset_type + '_performance_per_' + steps + '_lr' + str(
            args.lr) + '_' + heuristic +'_prune_' + str(prune) +  '_number_gcn_logsoftmax_' + '_' + val_dataset.__class__.__name__ + '_cuda' + str(
            args.cuda) + '.png')
    plt.clf()

def plot_loss_supervised(dataset_type='val', steps=None, t_plot = None, total_loss_val_np=None):

    plt.clf()
    plt.plot(t_plot, total_loss_val_np)
    # plt.legend(('loss'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
    #            loc='upper right')  # 'GNN-initial', 'GNN-RL', 'min-degree'
    plt.title(
        'Supervised loss curve ' + dataset_type + ' ' + val_dataset.__class__.__name__)
    plt.ylabel('loss')
    # plt.draw()
    plt.savefig(
        './results/supervised/'+heuristic+'/lr' + str(args.lr) + '/final_' + dataset_type + '_loss_per_' + steps + '_lr' + str(
            args.lr) + '_' + heuristic + '_prune_' + str(prune) + '_g2m_gcn_logsoftmax_' +
             '_' + val_dataset.__class__.__name__ + '_cuda' + str(
            args.cuda) + '.png')
    plt.clf()


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

dataset = UFSMDataset
if dataset.__name__ == 'UFSMDataset':
    with open('./data/UFSM/ss_small/ss_small.pkl', "rb") as f:
        val_ss_small = pkl.load(f)

    with open('./data/UFSM/ss_large/ss_large.pkl', "rb") as f:
         val_ss_large = pkl.load(f)

    # train_ss_large, val_ss_large, test_ss_large = open_dataset('./data/UFSM/ss_large/ss_large_split.pkl')

train_ER_small, val_ER_small, test_ER_small = open_dataset('./data/ERGcollection/erg_small.pkl')
train_ER_mid, val_ER_mid, test_ER_mid = open_dataset('./data/ERGcollection/erg_mid.pkl')

# build the GCN model
model = GCN_Sparse_Policy_SelectNode(nin=args.dinput,
                              nhidden= args.dhidden,
                              nout=args.doutput,
                              dropout=args.dropout,
                              ) # alpha=args.alpha
if args.cuda:
    model.cuda()

heuristic = 'min_degree' # 'one_step_greedy' 'min_degree'
prune = True

policy_sl = Train_SupervisedLearning(model=model, heuristic=heuristic,lr=args.lr, prune=prune, train_dataset=train_ER_small, val_dataset=val_ER_small, test_dataset=test_ER_small, use_cuda = args.cuda)



# Train the model


# total_loss_train = policy_sl.train(epochs=args.epochs, lr=args.lr)

val_dataset = val_ss_small

dataset_type = varname(val_ss_small)


# t_plot, total_loss_val_np, val_ave_gcn_np, val_ave_mind_np, val_ave_rand_np = policy_sl.validation_epochs(epochs=args.epochs, lr=args.lr, val_dataset=val_dataset, dataset_type=dataset_type)
#
# plot_performance_supervised(dataset_type=dataset_type,
#                             steps = 'epoch',
#                             t_plot=t_plot,
#                             val_ave_gcn_np=val_ave_gcn_np,
#                             val_ave_mind_np=val_ave_mind_np,
#                             val_ave_rand_np=val_ave_rand_np)
# plot_loss_supervised(dataset_type=dataset_type, steps='epoch', t_plot = t_plot, total_loss_val_np=total_loss_val_np)


steps_size=1000
steps_min=0
steps_max=41000 #1000000

t_plot, total_loss_val_np,val_ave_gcn_np, val_ave_mind_np, val_ave_rand_np = policy_sl.validation_steps(epochs=args.epochs, lr=args.lr, val_dataset=val_dataset, steps_min=steps_min, steps_max=steps_max, steps_size=steps_size, dataset_type=dataset_type)

# plot_performance_supervised(dataset_type=dataset_type,
#                             steps = str(steps_size)+'steps50000',
#                             t_plot=t_plot,
#                             val_ave_gcn_np=val_ave_gcn_np,
#                             val_ave_mind_np=val_ave_mind_np,
#                             val_ave_rand_np=val_ave_rand_np)
# plot_loss_supervised(dataset_type=dataset_type, steps=str(steps_size)+'steps50000', t_plot = t_plot, total_loss_val_np=total_loss_val_np)



#
# plot_performance_supervised(dataset='val',
#                             steps = '1000steps',
#                             t_plot=t_plot,
#                             val_ave_gcn_np=val_ave_gcn_np,
#                             val_ave_mind_np=val_ave_mind_np,
#                             val_ave_rand_np=val_ave_rand_np)
# plot_loss_supervised(dataset='val', steps='1000steps', t_plot = t_plot, total_loss_val_np=total_loss_val_np)

# plt.clf()
# plt.plot(t_plot, val_ave_gcn_np, t_plot, val_ave_mind_np, t_plot, val_ave_rand_np)
# plt.legend(('GNN', heuristic, 'random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#            loc='upper right')  # 'GNN-initial', 'GNN-RL', 'min-degree'
# plt.title(
#     'Performance on ' + val_dataset.__class__.__name__ + ' (average number of filled edges)')
# plt.ylabel('number of fill-in')
# # plt.draw()
# plt.savefig(
#     './results/supervised/validation_final_per_1000steps_lr' + str(
#         args.lr) + '_' + heuristic +'_prune_'+str(prune)+  '_perform_curve_g2m_number_gcn_logsoftmax_'+ varname(val_dataset) +'_' + val_dataset.__class__.__name__ + '_cuda' + str(
#         args.cuda) + '.png')
# plt.clf()
#
# plt.clf()
# plt.plot(t_plot, total_loss_val_np)
# # plt.legend(('loss'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
# #            loc='upper right')  # 'GNN-initial', 'GNN-RL', 'min-degree'
# plt.title(
#     'Supervised loss curve ' + val_dataset.__class__.__name__ )
# plt.ylabel('loss')
# # plt.draw()
# plt.savefig(
#     './results/supervised/validation_final_per_1000steps_lr' + str(
#         args.lr) + '_' + heuristic + '_prune_'+str(prune)+ '_loss_curve_logsoftmax_' + varname(val_dataset) +'_' + val_dataset.__class__.__name__ + '_cuda' + str(
#         args.cuda) + '.png')
# plt.clf()


# if args.cuda:
#     torch.save(model.state_dict(), './results/models/gcn_policy_'+heuristic+'_pre_'+dataset.__name__+str(args.nnode)+'dense_'+str(args.p)+'_epochs'+str(args.epochs)+'_cuda.pth')
#     # torch.save(model.state_dict(),
#     #            './results/models/gcn_policy_' + heuristic + '_pre_' + dataset.__name__ + '_epochs' + str(
#     #                args.epochs) + '_cuda.pth')
# else:
#     torch.save(model.state_dict(), './results/models/gcn_policy_min_degree_pre_erg100.pth')


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


