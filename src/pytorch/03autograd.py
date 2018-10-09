# encoding: utf-8
"""
@author: Dianlei Zhang
@contact: dianlei.zhang@qq.com

@time: 2018/10/9 14:35
@python version: 

"""
from torch.autograd import Variable
import torch

x = Variable(torch.ones(2, 2), requires_grad=True)
y = x + 2

z = y * y * 3
out = z.mean()
print("out:", out)

out.backward()
print("grad:", x.grad)
