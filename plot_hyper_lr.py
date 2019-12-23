import os
import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

plt.clf()
plt.figure(figsize=(12,5))

# val_set = ['ER_small_train','ER_small_val','ER_mid_val','ss_small_val','ss_large_val']
# step_size = [0.001,0.0005,0.0001,0.00001]
# steps_per_epoch = 40183
#
# heuristic = 'min_degree'
# time = 'step'
# performance = 'fill-in'
# loss = 'log kl loss'
# row_max = 40
# fz = 10
# t = np.zeros([5,row_max])
# loss_gcn = np.zeros([5,row_max])
# perf_gcn = np.zeros([5,row_max])
# perf_min = np.zeros([5,row_max])
# perf_ran = np.zeros([5,row_max])
#
#
# for i in range(len(step_size)):
#
#     t_file, loss_gcn_file, perf_min_file, perf_gcn_file, perf_ran_file = np.loadtxt('./results/logs/step_size/log_supervise_val_per_steps40000_nondet_'+str(step_size[i])+'_mindegree_without_prune_ER_small_train_logsoftmax_fulldepth.txt', delimiter=' ', usecols=(1,3,5,7,9), unpack=True)
#     if step_size[i] != 0.0001:
#         loss_gcn_file /= steps_per_epoch
#     t[i]= t_file[0:row_max]
#     loss_gcn[i] = loss_gcn_file[0:row_max]
#     loss_gcn[i] = np.log(loss_gcn[i])
#     perf_gcn[i] = perf_gcn_file[0:row_max]
#     perf_min[i] = perf_min_file[0:row_max]
#     perf_ran[i] = perf_ran_file[0:row_max]
#
#
# plt.clf()
# plt.figure(figsize=(12,10))
#
# ax1 = plt.subplot(221)
# plt.plot(t[0], loss_gcn[0], t[1], loss_gcn[1], t[2], loss_gcn[2], t[3], loss_gcn[3])
# plt.setp(ax1.get_xticklabels(), fontsize=fz)
# plt.xlabel(time,fontsize=fz)
# plt.ylabel(loss,fontsize=fz)
# plt.legend(('lr='+  str(step_size[0]),'lr='+str(step_size[1]),'lr='+str(step_size[2]),'lr='+str(step_size[3])),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#                loc='upper right',fontsize=fz)  # 'GNN-initial', 'GNN-RL', 'min-degree'
# plt.title('Averaged KL loss of ER_training set with different learning rate(lr)',fontsize=fz)
#
# ax2 = plt.subplot(222)
# plt.plot(t[0], perf_gcn[0], t[1], perf_gcn[1], t[2], perf_gcn[2], t[3], perf_gcn[3], t[3],perf_min[3], '--', t[3],perf_ran[3],'-.' )
# plt.setp(ax2.get_xticklabels(), fontsize=fz)
# plt.xlabel(time,fontsize=fz)
# plt.ylabel(performance,fontsize=fz)
# plt.legend(('lr='+  str(step_size[0]),'lr='+str(step_size[1]),'lr='+str(step_size[2]),'lr='+str(step_size[3]), 'min-degree','random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#                loc='upper right',fontsize=fz)  # 'GNN-initial', 'GNN-RL', 'min-degree'
# plt.title('Averaged fill-in of ER_training set with different learning rate(lr)',fontsize=fz)




step_size = [0.001,0.0005,0.0001,0.00001]
steps_per_epoch = 76709

heuristic = 'min_degree'
time = 'Number of $10^3$ steps'

performance = '$\hat{C}_{fillin}$'
loss = '$\hat{\mathcal{L}}_{KL}$ (log scale)'
# performance = 'Averaged fill-in'
# loss = 'Averaged KL loss (log scale)'
row_max = 40
fz = 12
t = np.zeros([5,row_max])
loss_gcn = np.zeros([5,row_max])
perf_gcn = np.zeros([5,row_max])
perf_min = np.zeros([5,row_max])
perf_ran = np.zeros([5,row_max])

dataset = 'ER_small_val'

for i in range(len(step_size)):

    t_file, loss_gcn_file, perf_min_file, perf_gcn_file, perf_ran_file = np.loadtxt('./results/logs/step_size/log_supervise_val_per_steps40000_nondet_'+str(step_size[i])+'_mindegree_without_prune_'+dataset+'_logsoftmax_fulldepth.txt', delimiter=' ', usecols=(1,3,5,7,9), unpack=True)

    loss_gcn_file /= steps_per_epoch
    t[i]= t_file[0:row_max]/1000
    loss_gcn[i] = loss_gcn_file[0:row_max]
    loss_gcn[i] = np.log(loss_gcn[i])
    perf_gcn[i] = perf_gcn_file[0:row_max]
    perf_min[i] = perf_min_file[0:row_max]
    perf_ran[i] = perf_ran_file[0:row_max]


# plt.clf()
# plt.figure(figsize=(12,5))

ax1 = plt.subplot(121)
plt.plot(t[0], loss_gcn[0], t[1], loss_gcn[1], t[2], loss_gcn[2], t[3], loss_gcn[3])
plt.setp(ax1.get_xticklabels(), fontsize=fz)
plt.xlabel(time,fontsize=fz)
plt.ylabel(loss,fontsize=fz)
plt.legend(('lr='+  str(step_size[0]),'lr='+str(step_size[1]),'lr='+str(step_size[2]),'lr='+str(step_size[3])),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
               loc='upper right',fontsize=fz)  # 'GNN-initial', 'GNN-RL', 'min-degree'
# plt.title('Averaged KL loss of validation set of ERs with different learning rate(lr)',fontsize=fz)

ax2 = plt.subplot(122)
plt.plot(t[0], perf_gcn[0], t[1], perf_gcn[1], t[2], perf_gcn[2], t[3], perf_gcn[3], t[3],perf_min[3], '--', t[3],perf_ran[3],'-.' )
plt.setp(ax2.get_xticklabels(), fontsize=fz)
plt.xlabel(time,fontsize=fz)
plt.ylabel(performance,fontsize=fz)
plt.legend(('lr='+  str(step_size[0]),'lr='+str(step_size[1]),'lr='+str(step_size[2]),'lr='+str(step_size[3]), 'min-degree','random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
               loc='upper right',fontsize=fz)  # 'GNN-initial', 'GNN-RL', 'min-degree'
# plt.title('Averaged fill-in of validation set of ERs with different learning rate(lr)',fontsize=fz)

plt.show()



# ax2 = plt.subplot(122)
# plt.plot(t[0], perf_gcn[0],t[0],perf_min[0], t[0],perf_ran[0])
# plt.setp(ax2.get_xticklabels(), fontsize=6)
# plt.xlabel(time)
# plt.ylabel(performance)
# plt.legend(('GNN', heuristic, 'random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#                loc='upper right')  # 'GNN-initial', 'GNN-RL', 'min-degree'
# plt.title(val_set[0])



# plt.xlim(1, 17.0)




# val_set = ['ER_small_train','ER_small_val','ER_mid_val','ss_small_val']
#
# heuristic = 'min_degree'
# time = 'steps'
# performance = 'fill-in'
# loss = 'kl loss'
# row_max = 40
#
# t = np.zeros([5,row_max])
# loss_gcn = np.zeros([5,row_max])
# perf_gcn = np.zeros([5,row_max])
# perf_min = np.zeros([5,row_max])
# perf_ran = np.zeros([5,row_max])
#
# for i in range(len(val_set)):
#
#     t_file, loss_gcn_file, perf_min_file, perf_gcn_file, perf_ran_file = np.loadtxt('./results/logs/val_steps/log_supervise_val_per_steps40000_nondet_0001_mindegree_without_prune_'+str(val_set[i])+'_logsoftmax_fulldepth.txt', delimiter=' ', usecols=(1,3,5,7,9), unpack=True)
#     t[i]= t_file[0:row_max]
#     loss_gcn[i] = loss_gcn_file[0:row_max]
#     perf_gcn[i] = perf_gcn_file[0:row_max]
#     perf_min[i] = perf_min_file[0:row_max]
#     perf_ran[i] = perf_ran_file[0:row_max]
#
#
# plt.clf()
# plt.figure(figsize=(30,30))
#
#
# ax1 = plt.subplot(421)
# plt.plot(t[0], loss_gcn[0])
# plt.setp(ax1.get_xticklabels(), fontsize=6)
# plt.xlabel(time)
# plt.ylabel(loss)
# plt.title(val_set[0])
#
#
# # share t only
# ax3 = plt.subplot(423, sharex=ax1)
# plt.plot(t[1],loss_gcn[1])
# plt.xlabel(time)
# plt.ylabel(loss)
# plt.title(val_set[1])
#
#
# # make these tick labels invisible
# plt.setp(ax3.get_xticklabels(), fontsize=6)
#
# ax5 = plt.subplot(425, sharex=ax1)
# plt.plot(t[2],loss_gcn[2])
# # make these tick labels invisible
# plt.setp(ax5.get_xticklabels(), fontsize=6)
# plt.xlabel(time)
# plt.ylabel(loss)
# plt.title(val_set[2])
#
#
# ax7 = plt.subplot(427, sharex=ax1)
# plt.plot(t[3],loss_gcn[3])
# # make these tick labels invisible
# plt.setp(ax7.get_xticklabels(),fontsize=6)
# plt.xlabel(time)
# plt.ylabel(loss)
# plt.title(val_set[3])
#
#
# # ax9 = plt.subplot(529, sharex=ax1)
# # plt.plot(t[4],loss_gcn[4])
# # # make these tick labels invisible
# # plt.setp(ax9.get_xticklabels())
# # plt.xlabel(time)
# # plt.ylabel(loss)
# # plt.title(val_set[4])
#
#
# ax2 = plt.subplot(422)
# plt.plot(t[0], perf_gcn[0],t[0],perf_min[0], t[0],perf_ran[0])
# plt.setp(ax2.get_xticklabels(), fontsize=6)
# plt.xlabel(time)
# plt.ylabel(performance)
# plt.legend(('GNN', heuristic, 'random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#                loc='upper right')  # 'GNN-initial', 'GNN-RL', 'min-degree'
# plt.title(val_set[0])
#
# # share t only
# ax4 = plt.subplot(424, sharex=ax1)
# plt.plot(t[1],perf_gcn[1],t[1],perf_min[1],t[1], perf_ran[1])
# plt.xlabel(time)
# plt.ylabel(performance)
# plt.legend(('GNN', heuristic, 'random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#                loc='upper right')  # 'GNN-initial', 'GNN-RL', 'min-degree'
# # make these tick labels invisible
# plt.setp(ax4.get_xticklabels(), fontsize=6)
# plt.title(val_set[1])
#
# ax6 = plt.subplot(426, sharex=ax1)
# plt.plot(t[2],perf_gcn[2],t[2],perf_min[2],t[2], perf_ran[2])
# # make these tick labels invisible
# plt.xlabel(time)
# plt.ylabel(performance)
# plt.legend(('GNN', heuristic, 'random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#                loc='upper right')  # 'GNN-initial', 'GNN-RL', 'min-degree'
# plt.setp(ax6.get_xticklabels(), fontsize=6)
# plt.title(val_set[2])
#
# ax8 = plt.subplot(428, sharex=ax1)
# plt.plot(t[3],perf_gcn[3],t[3],perf_min[3],t[3], perf_ran[3])
# # make these tick labels invisible
# plt.xlabel(time)
# plt.ylabel(performance)
# plt.legend(('GNN', heuristic, 'random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#                loc='upper right')  # 'GNN-initial', 'GNN-RL', 'min-degree'
# plt.setp(ax8.get_xticklabels(),fontsize=6)
# plt.title(val_set[3])
#
#
# # ax10 = plt.subplot(5,2,10, sharex=ax1)
# # plt.plot(t[4],perf_gcn[4],t[4],perf_min[4],t[4], perf_ran[4])
# # # make these tick labels invisible
# # plt.xlabel(time)
# # plt.ylabel(performance)
# # plt.legend(('GNN', heuristic, 'random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
# #                loc='upper right')  # 'GNN-initial', 'GNN-RL', 'min-degree'
# # plt.setp(ax10.get_xticklabels())
# # plt.title(val_set[4])
#
#
# # plt.xlim(1, 17.0)
#
# plt.show()
#



