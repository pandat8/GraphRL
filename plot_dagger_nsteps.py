import os
import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


val_set = ['ER_small_train','ss_small_val']

heuristic = 'min_degree'
time = 'Number of training data'
performance = 'fill-in'
loss = 'log kl loss'

row_max = 40
fz = 10
t = np.zeros([5,row_max])
loss_gcn = np.zeros([5,row_max])
perf_gcn = np.zeros([5,row_max])
perf_min = np.zeros([5,row_max])
perf_ran = np.zeros([5,row_max])

plt.clf()
plt.figure(figsize=(12,10))
# plt.figure(figsize=(16,18))
for i in range(2):

    t_file, loss_gcn_file, perf_min_file, perf_gcn_file, perf_ran_file = np.loadtxt('./results/logs/dagger_nsteps/log_supervise_val_per_steps40000_nondet_0001_mindegree_without_prune_'+str(val_set[i])+'_logsoftmax_fulldepth.txt', delimiter=' ', usecols=(1,3,5,7,9), unpack=True)
    t[i]= t_file[0:row_max]
    loss_gcn[i] = loss_gcn_file[0:row_max]
    loss_gcn[i] = np.log(loss_gcn[i])
    perf_gcn[i] = perf_gcn_file[0:row_max]
    perf_min[i] = perf_min_file[0:row_max]
    perf_ran[i] = perf_ran_file[0:row_max]


val_set = ['ER_small_train','ss_small_val']

for i in range(2,4,1):

    t_file, loss_gcn_file, perf_min_file, perf_gcn_file, perf_ran_file = np.loadtxt('./results/logs/dagger_nsteps/log_dagger_supervise_val_per_steps40000_nondet_0001_mindegree_without_prune_'+str(val_set[i-2])+'_logsoftmax_fulldepth.txt', delimiter=' ', usecols=(1,3,5,7,9), unpack=True)
    t[i]= t_file[0:row_max]
    loss_gcn[i] = loss_gcn_file[0:row_max]
    loss_gcn[i] = np.log(loss_gcn[i])
    perf_gcn[i] = perf_gcn_file[0:row_max]
    perf_min[i] = perf_min_file[0:row_max]
    perf_ran[i] = perf_ran_file[0:row_max]


val_set = ['ER_small_train','SSMC_small']

ax1 = plt.subplot(221)
plt.plot(t[0], loss_gcn[0], t[2], loss_gcn[2])
plt.xlabel(time,fontsize=fz)
plt.ylabel(loss, fontsize=fz)
plt.setp(ax1.get_xticklabels(), fontsize=fz)
plt.setp(ax1.get_yticklabels(), fontsize=fz)
plt.legend(('GNN', 'Dagger'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
               loc='upper right', fontsize=fz)  # 'GNN-initial', 'GNN-RL', 'min-degree'
plt.title(val_set[0], fontsize=fz)


# share t only
ax3 = plt.subplot(223, sharex=ax1)
plt.plot(t[1],loss_gcn[1], t[3], loss_gcn[3])
plt.xlabel(time, fontsize=fz)
plt.ylabel(loss, fontsize=fz)
plt.legend(('GNN', 'Dagger'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
               loc='upper right', fontsize=fz)  # 'GNN-initial', 'GNN-RL', 'min-degree'
plt.title(val_set[1], fontsize=fz)
# make these tick labels invisible
plt.setp(ax3.get_xticklabels(), fontsize=fz)
plt.setp(ax3.get_yticklabels(), fontsize=fz)

# ax5 = plt.subplot(525, sharex=ax1)
# plt.plot(t[2],loss_gcn[2])
# # make these tick labels invisible
# plt.setp(ax5.get_xticklabels(), fontsize=16)
# plt.setp(ax5.get_yticklabels(), fontsize=16)
# plt.xlabel(time, fontsize=20)
# plt.ylabel(loss, fontsize=20)
# plt.title(val_set[2], fontsize=20)
#
#
# ax7 = plt.subplot(527, sharex=ax1)
# plt.plot(t[3],loss_gcn[3])
# # make these tick labels invisible
# plt.setp(ax7.get_xticklabels(), fontsize=16)
# plt.setp(ax7.get_yticklabels(), fontsize=16)
# plt.xlabel(time, fontsize=20)
# plt.ylabel(loss, fontsize=20)
# plt.title(val_set[3], fontsize=20)
#
#
# ax9 = plt.subplot(529, sharex=ax1)
# plt.plot(t[4],loss_gcn[4])
# # make these tick labels invisible
# plt.setp(ax9.get_xticklabels(), fontsize=16)
# plt.setp(ax9.get_yticklabels(), fontsize=16)
# plt.xlabel(time, fontsize=20)
# plt.ylabel(loss, fontsize=20)
# plt.title(val_set[4], fontsize=20)


ax2 = plt.subplot(222)
plt.plot(t[0], perf_gcn[0],t[0],perf_min[0], t[0],perf_ran[0],t[2], perf_gcn[2])
plt.setp(ax2.get_xticklabels(), fontsize=fz)
plt.setp(ax2.get_yticklabels(), fontsize=fz)
plt.xlabel(time, fontsize=fz)
plt.ylabel(performance, fontsize=fz)
plt.legend(('GNN', heuristic, 'random','Dagger'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
               loc='upper right', fontsize=fz)  # 'GNN-initial', 'GNN-RL', 'min-degree'
plt.title(val_set[0], fontsize=fz)

# share t only
ax4 = plt.subplot(224, sharex=ax1)
plt.plot(t[1],perf_gcn[1],t[1],perf_min[1],t[1], perf_ran[1], t[3],perf_gcn[3])
plt.xlabel(time, fontsize=fz)
plt.ylabel(performance, fontsize=fz)
plt.legend(('GNN', heuristic, 'random','Dagger'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
               loc='upper right', fontsize=fz)  # 'GNN-initial', 'GNN-RL', 'min-degree'
# make these tick labels invisible
plt.setp(ax4.get_xticklabels(), fontsize=fz)
plt.setp(ax4.get_yticklabels(), fontsize=fz)
plt.title(val_set[1], fontsize=fz)

# ax6 = plt.subplot(526, sharex=ax1)
# plt.plot(t[2],perf_gcn[2],t[2],perf_min[2],t[2], perf_ran[2])
# # make these tick labels invisible
# plt.xlabel(time, fontsize=20)
# plt.ylabel(performance, fontsize=20)
# plt.legend(('GNN', heuristic, 'random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#                loc='upper right', fontsize=20)  # 'GNN-initial', 'GNN-RL', 'min-degree'
# plt.setp(ax6.get_xticklabels(), fontsize=16)
# plt.setp(ax6.get_yticklabels(), fontsize=16)
# plt.title(val_set[2], fontsize=20)
#
# ax8 = plt.subplot(528, sharex=ax1)
# plt.plot(t[3],perf_gcn[3],t[3],perf_min[3],t[3], perf_ran[3])
# # make these tick labels invisible
# plt.xlabel(time, fontsize=20)
# plt.ylabel(performance, fontsize=20)
# plt.legend(('GNN', heuristic, 'random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#                loc='upper right', fontsize=20)  # 'GNN-initial', 'GNN-RL', 'min-degree'
# plt.setp(ax8.get_xticklabels(), fontsize=16)
# plt.setp(ax8.get_yticklabels(), fontsize=16)
# plt.title(val_set[3], fontsize=20)
#
#
# ax10 = plt.subplot(5,2,10, sharex=ax1)
# plt.plot(t[4],perf_gcn[4],t[4],perf_min[4],t[4], perf_ran[4])
# # make these tick labels invisible
# plt.xlabel(time, fontsize=20)
# plt.ylabel(performance, fontsize=20)
# plt.legend(('GNN', heuristic, 'random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#                loc='upper right', fontsize=20)  # 'GNN-initial', 'GNN-RL', 'min-degree'
# plt.setp(ax10.get_xticklabels(), fontsize=16)
# plt.setp(ax10.get_yticklabels(), fontsize=16)
# plt.title(val_set[4], fontsize=20)
#



plt.show()


val_set = ['ER_small_train','ss_small_val']

heuristic = 'min_degree'
time = 'epochs'
performance = 'fill-in'
loss = 'log kl loss'

row_max = 21
fz = 10
t = np.zeros([5,row_max])
loss_gcn = np.zeros([5,row_max])
perf_gcn = np.zeros([5,row_max])
perf_min = np.zeros([5,row_max])
perf_ran = np.zeros([5,row_max])

plt.clf()
plt.figure(figsize=(12,10))
# plt.figure(figsize=(16,18))
for i in range(2):

    t_file, loss_gcn_file, perf_min_file, perf_gcn_file, perf_ran_file = np.loadtxt('./results/logs/dagger_nsteps/log_supervise_val_per_epochs20_nondet_0001_mindegree_without_prune_'+str(val_set[i])+'_logsoftmax_fulldepth.txt', delimiter=' ', usecols=(1,3,5,7,9), unpack=True)
    t[i]= t_file[0:row_max]
    loss_gcn[i] = loss_gcn_file[0:row_max]
    loss_gcn[i] = np.log(loss_gcn[i])
    perf_gcn[i] = perf_gcn_file[0:row_max]
    perf_min[i] = perf_min_file[0:row_max]
    perf_ran[i] = perf_ran_file[0:row_max]


val_set = ['ER_small_train','ss_small_val']

for i in range(2,4,1):

    t_file, loss_gcn_file, perf_min_file, perf_gcn_file, perf_ran_file = np.loadtxt('./results/logs/dagger_nsteps/log_dagger_supervise_val_per_epochs20_nondet_0001_mindegree_without_prune_'+str(val_set[i-2])+'_logsoftmax_fulldepth.txt', delimiter=' ', usecols=(1,3,5,7,9), unpack=True)
    t[i]= t_file[0:row_max]
    loss_gcn[i] = loss_gcn_file[0:row_max]
    loss_gcn[i] = np.log(loss_gcn[i])
    perf_gcn[i] = perf_gcn_file[0:row_max]
    perf_min[i] = perf_min_file[0:row_max]
    perf_ran[i] = perf_ran_file[0:row_max]


val_set = ['ER_small_train','SSMC_small']

ax1 = plt.subplot(221)
plt.plot(t[0], loss_gcn[0], t[2], loss_gcn[2])
plt.xlabel(time,fontsize=fz)
plt.ylabel(loss, fontsize=fz)
plt.setp(ax1.get_xticklabels(), fontsize=fz)
plt.setp(ax1.get_yticklabels(), fontsize=fz)
plt.legend(('GNN', 'Dagger'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
               loc='upper right', fontsize=fz)  # 'GNN-initial', 'GNN-RL', 'min-degree'
plt.title(val_set[0], fontsize=fz)


# share t only
ax3 = plt.subplot(223, sharex=ax1)
plt.plot(t[1],loss_gcn[1], t[3], loss_gcn[3])
plt.xlabel(time, fontsize=fz)
plt.ylabel(loss, fontsize=fz)
plt.legend(('GNN', 'Dagger'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
               loc='upper right', fontsize=fz)  # 'GNN-initial', 'GNN-RL', 'min-degree'
plt.title(val_set[1], fontsize=fz)
# make these tick labels invisible
plt.setp(ax3.get_xticklabels(), fontsize=fz)
plt.setp(ax3.get_yticklabels(), fontsize=fz)

# ax5 = plt.subplot(525, sharex=ax1)
# plt.plot(t[2],loss_gcn[2])
# # make these tick labels invisible
# plt.setp(ax5.get_xticklabels(), fontsize=16)
# plt.setp(ax5.get_yticklabels(), fontsize=16)
# plt.xlabel(time, fontsize=20)
# plt.ylabel(loss, fontsize=20)
# plt.title(val_set[2], fontsize=20)
#
#
# ax7 = plt.subplot(527, sharex=ax1)
# plt.plot(t[3],loss_gcn[3])
# # make these tick labels invisible
# plt.setp(ax7.get_xticklabels(), fontsize=16)
# plt.setp(ax7.get_yticklabels(), fontsize=16)
# plt.xlabel(time, fontsize=20)
# plt.ylabel(loss, fontsize=20)
# plt.title(val_set[3], fontsize=20)
#
#
# ax9 = plt.subplot(529, sharex=ax1)
# plt.plot(t[4],loss_gcn[4])
# # make these tick labels invisible
# plt.setp(ax9.get_xticklabels(), fontsize=16)
# plt.setp(ax9.get_yticklabels(), fontsize=16)
# plt.xlabel(time, fontsize=20)
# plt.ylabel(loss, fontsize=20)
# plt.title(val_set[4], fontsize=20)


ax2 = plt.subplot(222)
plt.plot(t[0], perf_gcn[0],t[0],perf_min[0], t[0],perf_ran[0],t[2], perf_gcn[2])
plt.setp(ax2.get_xticklabels(), fontsize=fz)
plt.setp(ax2.get_yticklabels(), fontsize=fz)
plt.xlabel(time, fontsize=fz)
plt.ylabel(performance, fontsize=fz)
plt.legend(('GNN', heuristic, 'random','Dagger'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
               loc='upper right', fontsize=fz)  # 'GNN-initial', 'GNN-RL', 'min-degree'
plt.title(val_set[0], fontsize=fz)

# share t only
ax4 = plt.subplot(224, sharex=ax1)
plt.plot(t[1],perf_gcn[1],t[1],perf_min[1],t[1], perf_ran[1], t[3],perf_gcn[3])
plt.xlabel(time, fontsize=fz)
plt.ylabel(performance, fontsize=fz)
plt.legend(('GNN', heuristic, 'random','Dagger'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
               loc='upper right', fontsize=fz)  # 'GNN-initial', 'GNN-RL', 'min-degree'
# make these tick labels invisible
plt.setp(ax4.get_xticklabels(), fontsize=fz)
plt.setp(ax4.get_yticklabels(), fontsize=fz)
plt.title(val_set[1], fontsize=fz)

# ax6 = plt.subplot(526, sharex=ax1)
# plt.plot(t[2],perf_gcn[2],t[2],perf_min[2],t[2], perf_ran[2])
# # make these tick labels invisible
# plt.xlabel(time, fontsize=20)
# plt.ylabel(performance, fontsize=20)
# plt.legend(('GNN', heuristic, 'random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#                loc='upper right', fontsize=20)  # 'GNN-initial', 'GNN-RL', 'min-degree'
# plt.setp(ax6.get_xticklabels(), fontsize=16)
# plt.setp(ax6.get_yticklabels(), fontsize=16)
# plt.title(val_set[2], fontsize=20)
#
# ax8 = plt.subplot(528, sharex=ax1)
# plt.plot(t[3],perf_gcn[3],t[3],perf_min[3],t[3], perf_ran[3])
# # make these tick labels invisible
# plt.xlabel(time, fontsize=20)
# plt.ylabel(performance, fontsize=20)
# plt.legend(('GNN', heuristic, 'random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#                loc='upper right', fontsize=20)  # 'GNN-initial', 'GNN-RL', 'min-degree'
# plt.setp(ax8.get_xticklabels(), fontsize=16)
# plt.setp(ax8.get_yticklabels(), fontsize=16)
# plt.title(val_set[3], fontsize=20)
#
#
# ax10 = plt.subplot(5,2,10, sharex=ax1)
# plt.plot(t[4],perf_gcn[4],t[4],perf_min[4],t[4], perf_ran[4])
# # make these tick labels invisible
# plt.xlabel(time, fontsize=20)
# plt.ylabel(performance, fontsize=20)
# plt.legend(('GNN', heuristic, 'random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#                loc='upper right', fontsize=20)  # 'GNN-initial', 'GNN-RL', 'min-degree'
# plt.setp(ax10.get_xticklabels(), fontsize=16)
# plt.setp(ax10.get_yticklabels(), fontsize=16)
# plt.title(val_set[4], fontsize=20)
#
plt.show()


# val_set = ['ER_small_train','ER_small_val','ER_mid_val','ss_small_val', 'ss_large_val']
#
# heuristic = 'min_degree'
# time = '$10^3$ steps'
# performance = 'fill-in'
# loss = 'log kl loss'
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
#     t[i]= t_file[0:row_max]/1000
#     loss_gcn[i] = loss_gcn_file[0:row_max]
#     loss_gcn[i] = np.log(loss_gcn[i])
#     perf_gcn[i] = perf_gcn_file[0:row_max]
#     perf_min[i] = perf_min_file[0:row_max]
#     perf_ran[i] = perf_ran_file[0:row_max]
#
#
# val_set = ['ER_small_train','ER_small_val','ER_large','SSMC_small','SSMC_large']
#
# plt.clf()
# plt.figure(figsize=(16,18))
#
#
# ax1 = plt.subplot(521)
# plt.plot(t[0], loss_gcn[0])
# plt.setp(ax1.get_xticklabels(), fontsize=16)
# plt.setp(ax1.get_yticklabels(), fontsize=16)
# plt.xlabel(time, fontsize=20)
# plt.ylabel(loss, fontsize=20)
# plt.title(val_set[0], fontsize=20)
#
#
# # share t only
# ax3 = plt.subplot(523, sharex=ax1)
# plt.plot(t[1],loss_gcn[1])
# plt.xlabel(time, fontsize=20)
# plt.ylabel(loss, fontsize=20)
# plt.title(val_set[1], fontsize=20)
# # make these tick labels invisible
# plt.setp(ax3.get_xticklabels(), fontsize=16)
# plt.setp(ax3.get_yticklabels(), fontsize=16)
#
# ax5 = plt.subplot(525, sharex=ax1)
# plt.plot(t[2],loss_gcn[2])
# # make these tick labels invisible
# plt.setp(ax5.get_xticklabels(), fontsize=16)
# plt.setp(ax5.get_yticklabels(), fontsize=16)
# plt.xlabel(time, fontsize=20)
# plt.ylabel(loss, fontsize=20)
# plt.title(val_set[2], fontsize=20)
#
#
# ax7 = plt.subplot(527, sharex=ax1)
# plt.plot(t[3],loss_gcn[3])
# # make these tick labels invisible
# plt.setp(ax7.get_xticklabels(), fontsize=16)
# plt.setp(ax7.get_yticklabels(), fontsize=16)
# plt.xlabel(time, fontsize=20)
# plt.ylabel(loss, fontsize=20)
# plt.title(val_set[3], fontsize=20)
#
#
# ax9 = plt.subplot(529, sharex=ax1)
# plt.plot(t[4],loss_gcn[4])
# # make these tick labels invisible
# plt.setp(ax9.get_xticklabels(), fontsize=16)
# plt.setp(ax9.get_yticklabels(), fontsize=16)
# plt.xlabel(time, fontsize=20)
# plt.ylabel(loss, fontsize=20)
# plt.title(val_set[4], fontsize=20)
#
#
# ax2 = plt.subplot(522)
# plt.plot(t[0], perf_gcn[0],t[0],perf_min[0], t[0],perf_ran[0])
# plt.setp(ax2.get_xticklabels(), fontsize=16)
# plt.setp(ax2.get_yticklabels(), fontsize=16)
# plt.xlabel(time, fontsize=20)
# plt.ylabel(performance, fontsize=20)
# plt.legend(('GNN', heuristic, 'random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#                loc='upper right', fontsize=20)  # 'GNN-initial', 'GNN-RL', 'min-degree'
# plt.title(val_set[0], fontsize=20)
#
# # share t only
# ax4 = plt.subplot(524, sharex=ax1)
# plt.plot(t[1],perf_gcn[1],t[1],perf_min[1],t[1], perf_ran[1])
# plt.xlabel(time, fontsize=20)
# plt.ylabel(performance, fontsize=20)
# plt.legend(('GNN', heuristic, 'random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#                loc='upper right', fontsize=20)  # 'GNN-initial', 'GNN-RL', 'min-degree'
# # make these tick labels invisible
# plt.setp(ax4.get_xticklabels(), fontsize=16)
# plt.setp(ax4.get_yticklabels(), fontsize=16)
# plt.title(val_set[1], fontsize=20)
#
# ax6 = plt.subplot(526, sharex=ax1)
# plt.plot(t[2],perf_gcn[2],t[2],perf_min[2],t[2], perf_ran[2])
# # make these tick labels invisible
# plt.xlabel(time, fontsize=20)
# plt.ylabel(performance, fontsize=20)
# plt.legend(('GNN', heuristic, 'random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#                loc='upper right', fontsize=20)  # 'GNN-initial', 'GNN-RL', 'min-degree'
# plt.setp(ax6.get_xticklabels(), fontsize=16)
# plt.setp(ax6.get_yticklabels(), fontsize=16)
# plt.title(val_set[2], fontsize=20)
#
# ax8 = plt.subplot(528, sharex=ax1)
# plt.plot(t[3],perf_gcn[3],t[3],perf_min[3],t[3], perf_ran[3])
# # make these tick labels invisible
# plt.xlabel(time, fontsize=20)
# plt.ylabel(performance, fontsize=20)
# plt.legend(('GNN', heuristic, 'random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#                loc='upper right', fontsize=20)  # 'GNN-initial', 'GNN-RL', 'min-degree'
# plt.setp(ax8.get_xticklabels() , fontsize=16)
# plt.setp(ax8.get_yticklabels() , fontsize=16)
# plt.title(val_set[3], fontsize=20)
#
#
# ax10 = plt.subplot(5,2,10, sharex=ax1)
# plt.plot(t[4],perf_gcn[4],t[4],perf_min[4],t[4], perf_ran[4])
# # make these tick labels invisible
# plt.xlabel(time, fontsize=20)
# plt.ylabel(performance, fontsize=20)
# plt.legend(('GNN', heuristic, 'random'),  # 'GNN-RL', 'GNN-RL-epsilon', 'min-degree'
#                loc='upper right')  # 'GNN-initial', 'GNN-RL', 'min-degree'
# plt.setp(ax10.get_xticklabels(), fontsize=16)
# plt.setp(ax10.get_yticklabels(), fontsize=16)
# plt.title(val_set[4],fontsize=20)
#
#
# # plt.xticks(np.arange(min(t[0]), max(t[0]), 1.0))
#
# plt.show()
#
#
#
#
