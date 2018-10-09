# encoding: utf-8
"""
@author: Dianlei Zhang
@contact: dianlei.zhang@qq.com

@time: 2018/10/9 14:24
@python version: 

"""
import numpy as np
import torch

# torch 与 numpy相互转换
a = torch.ones(5)
b = a.numpy()
print(a, b)
a.add_(1)
print(a, b)

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a, b)



