import tarfile
import os
import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


l, p = np.loadtxt('./results/logs/log_supervise_gridsearch_mindegree__ER_small_20graphs_train.txt', delimiter=' ', usecols=(7, 9), unpack=True)
l= np.reshape(l,(41,41))
print(l)
p= np.reshape(p,(41,41))

w1_all = np.linspace(-5,5,41)
w2_all = np.linspace(-5,5,41)

# Z = []
for w2 in range(-20,21,1):
    w2 /= 4

    for w1 in range(-20, 21, 1):
        w1 /= 4

        # z = (1 - w1 / 2 + w1 ** 5 + w2 ** 3) * np.exp(-w1 ** 2 - w2 ** 2)
        # Z.append(z)


plt.close()
W1, W2 = np.meshgrid(w1_all, w2_all)

p=p/12593.35
l=np.log(l)


W1 = W1[0:40,12:29]
W2 = W2[0:40,12:29]
l =  l[0:40,12:29]
p =  p[0:40,12:29]

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(W1, W2,l ,rstride=1, cstride=1, cmap='rainbow')

# ax.set_zlim(0.9, 1.30)
# ax.set_ybound(-5,0)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=8)
# print(ax.azim)
# ax.view_init(azim=-90)

plt.xlabel('W1')
plt.ylabel('W2')

plt.show()


fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(W1, W2,p, rstride=1, cstride=1, cmap='rainbow'
                       )
# cmap=cm.coolwarm, linewidth=0, antialiased=False

# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

# ax.set_zlim(0.9, 1.30)
# ax.set_ylim(-5,-1)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=8)
plt.xlabel('W1')
plt.ylabel('W2')

plt.show()

W1 = W1[0:28,4:13]
W2 = W2[0:28,4:13]
l =  l[0:28,4:13]
p =  p[0:28,4:13]



plt.clf()
fig = plt.figure()
Cf = plt.contourf(W1, W2, l, 5, levels=[ np.log(i/10000) for i in range(1,50,10)], alpha=.75, cmap=plt.cm.hot, extend='both')
C = plt.contour(W1, W2, l,5,levels=[ np.log(i/10000) for i in range(1,50,10)], colors='black', linewidth=.5, extend='both')
plt.clabel(C, inline=1, fontsize=9)
fig.colorbar(Cf, shrink=0.5, aspect=8)
plt.xlabel('W1')
plt.ylabel('W2')
plt.show()

plt.clf()

plt.clf()
fig = plt.figure()
Cf = plt.contourf(W1, W2, p,7, levels=[ 1.00001,1.001,1.05,1.1,1.2,1.3], alpha=.75, cmap=plt.cm.hot,  extend='both') # levels=[ i/100 for i in range(100,110,1)],
C = plt.contour(W1, W2, p,7, levels=[1.00001,1.001,1.05,1.1,1.2,1.3], colors='black', linewidth=.5, extend='both')
fig.colorbar(Cf, shrink=0.5, aspect=8)
plt.clabel(C, inline=1, fontsize=9)
plt.xlabel('W1')
plt.ylabel('W2')
plt.show()







# def untar(input_dir, output_dir):
#     for path, directories, files in os.walk(input_dir):
#         for f in files:
#
#             if f.endswith(".tar.gz"):
#                 index_of_dot = f.index('.')
#                 f_name_without_extension = f[:index_of_dot]
#
#                 tar = tarfile.open(os.path.join(path, f), 'r:gz')
#                 for member in tar.getmembers():
#
#                     if member.name.endswith(f_name_without_extension+'.mtx'):  # skip if the TarInfo is not files
#                         member.name = os.path.basename(member.name)  # remove the path by reset it
#                         tar.extract(member, output_dir)  # extract
#                 tar.close()
#
#
#
# if __name__ == '__main__':
#     untar('./data/UFSM/ss_large/ss_large_source/', './data/UFSM/ss_large/ss_large_set/')
