import numpy as np
import argparse
import torch
from collections import OrderedDict

import matplotlib.pyplot as plt
import inspect, re
import pickle as pkl

from data.ergDataset import ErgDataset
from utils.utils import open_dataset, varname

# from data.UFSMDataset import UFSMDataset
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

# def plot_performance_supervised(dataset = 'val',steps=None, t_plot=None, val_ave_gcn_np=None, val_ave_mind_np=None, val_ave_rand_np=None):
#
#     plt.clf()
#     plt.plot(t_plot, val_ave_gcn_np, t_plot, val_ave_mind_np, t_plot, val_ave_rand_np)
#     plt.legend(('GNN', heuristic, 'random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#                loc='upper right')  # 'GNN-initial', 'GNN-RL', 'min-degree'
#     plt.title(
#         'Performance on ' + dataset + ' '+ val_dataset.__class__.__name__ + ' (average number of filled edges)')
#     plt.ylabel('number of fill-in')
#     # plt.draw()
#     plt.savefig(
#         './results/supervised/final_'+dataset+'_performance_per_'+steps+'_lr' + str(
#             args.lr) + '_' + heuristic +'_prune_'+str(prune)+  '_number_gcn_logsoftmax_'+ '_' + val_dataset.__class__.__name__ + '_cuda' + str(
#             args.cuda) + '.png')
#     plt.clf()
#
# def plot_loss_supervised(dataset='val', steps=None, t_plot = None, total_loss_val_np=None):
#
#     plt.clf()
#     plt.plot(t_plot, total_loss_val_np)
#     # plt.legend(('loss'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#     #            loc='upper right')  # 'GNN-initial', 'GNN-RL', 'min-degree'
#     plt.title(
#         'Supervised loss curve ' + dataset + ' '+ val_dataset.__class__.__name__)
#     plt.ylabel('loss')
#     # plt.draw()
#     plt.savefig(
#         './results/supervised/final_'+dataset+'_loss_per_'+steps+'_lr' + str(
#             args.lr) + '_' + heuristic + '_prune_' + str(prune) + '_g2m_gcn_logsoftmax_'+
#              '_' + val_dataset.__class__.__name__ + '_cuda' + str(
#             args.cuda) + '.png')
#     plt.clf()



dataset = ErgDataset
if dataset.__name__ == 'UFSMDataset':
    train_dataset = dataset(start=18, end=19)
    val_dataset = dataset(start=19, end=20)
    test_dataset = dataset(start=24, end=26)

elif dataset.__name__ == 'ErgDataset':
    train_ER_small, val_ER_small, test_ER_small = open_dataset('./data/ERGcollection/erg_small.pkl')

# build the GCN model
model = GCN_Sparse_Policy_SelectNode(nin=args.dinput,
                              nhidden= args.dhidden,
                              nout=args.doutput,
                              dropout=args.dropout,
                              ) # alpha=args.alpha
if args.cuda:
    model.cuda()
print(model.state_dict())

heuristic = 'min_degree' # 'one_step_greedy' 'min_degree'
prune = True

policy_sl = Train_SupervisedLearning(model=model,
                                     heuristic=heuristic,
                                     lr=args.lr,
                                     prune=prune,
                                     train_dataset=train_ER_small,
                                     val_dataset=val_ER_small,
                                     test_dataset=test_ER_small,
                                     use_cuda = args.cuda)


val_dataset = train_ER_small

dataset_type = varname(train_ER_small)

policy_sl.validation_gridsearch(val_dataset = val_dataset, dataset_type=dataset_type)

# w1_all = np.linspace(-5,5,41)
# w2_all = np.linspace(-5,5,41)
# print(w1_all)
# Z = []
# for w2 in range(-20,21,1):
#     w2 /= 4
#
#     for w1 in range(-20, 21, 1):
#         w1 /= 4
#
#         z = (1 - w1 / 2 + w1 ** 5 + w2 ** 3) * np.exp(-w1 ** 2 - w2 ** 2)
#         Z.append(z)
#
# print(Z)
# Z_np = np.array(Z).reshape(41,41)
#
# W1, W2 = np.meshgrid(w1_all, w2_all)
# # Z = (1-W1/2+W1**5+W2**3)*np.exp(-W1**2-W2**2)
# plt.clf()
# plt.contourf(W1, W2, Z_np,16, alpha=.75, cmap=plt.cm.hot)
# C = plt.contour(W1, W2, Z_np,16, colors='black', linewidth=.5)
# plt.clabel(C, inline=1, fontsize=10)
# plt.show()

        # new_state_dict = OrderedDict({'gc1.weight': torch.Tensor([[w1]]), 'gc2.weight': torch.Tensor([[w2]])})
        # model.load_state_dict(new_state_dict, strict=False)




#
#
# print(model.state_dict())

# model.load_state_dict(new_state_dict, strict=False)
#
# for name, param in model.named_parameters():
#
#                 print('parameter name {}'.format(name),
#                     'parameter value {}'.format(param.data))
#                 print(type(name))
#                 print(param.data)
#                 param.data = torch.Tensor([[2]])
#                 print('parameter name {}'.format(name),
#                       'parameter value {}'.format(param.data))

# heuristic = 'min_degree' # 'one_step_greedy' 'min_degree'
# prune = False
# policy_sl = Train_SupervisedLearning(model=model, heuristic=heuristic,lr=args.lr, prune=prune, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, use_cuda = args.cuda)



# Train the model


# total_loss_train = policy_sl.train(epochs=args.epochs, lr=args.lr)

# val_dataset = val_dataset
# t_plot, total_loss_val_np, val_ave_gcn_np, val_ave_mind_np, val_ave_rand_np = policy_sl.validation(epochs=args.epochs, lr=args.lr, val_dataset=train_dataset)
#
# plot_performance_supervised(dataset='train',
#                             steps = 'epoch',
#                             t_plot=t_plot,
#                             val_ave_gcn_np=val_ave_gcn_np,
#                             val_ave_mind_np=val_ave_mind_np,
#                             val_ave_rand_np=val_ave_rand_np)
# plot_loss_supervised(dataset='train', steps='epoch', t_plot = t_plot, total_loss_val_np=total_loss_val_np)
#


# t_plot, total_loss_val_np,val_ave_gcn_np, val_ave_mind_np, val_ave_rand_np = policy_sl.validation_steps(epochs=args.epochs, lr=args.lr, val_dataset=train_dataset, steps_max=164800)
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


