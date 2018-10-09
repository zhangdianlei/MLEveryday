# encoding: utf-8
"""
@author: Dianlei Zhang
@contact: dianlei.zhang@qq.com

@time: 2018/10/9 14:00
@python version: 

"""

from __future__ import print_function
import torch

x = torch.Tensor(5, 3)
y = torch.rand(5, 3)

print(x, "\n", y)

print("y", torch.add(x, y))

y.add_(x)

print("y", y)
