import numpy as np
t1 = np.arange(0,10,1)
t2 = np.arange(10,0,-1)
t3 = np.array([[1,2,3],[4,5,6]])
t4 = np.copy(t3)
t3[0,0] = 0
print(t4)
