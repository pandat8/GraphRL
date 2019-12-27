import numpy as np
import argparse
import torch

# for i in range(18, 19):
#     print(i)
#     print(i+1)

a = torch.tensor([[1,2],[3,4]])
b = torch.tensor([[1,1]])
print(a.size)
print(b.size)
print(a+b)


