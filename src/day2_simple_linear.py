# encoding: utf-8
"""
@author: Dianlei Zhang
@contact: dianlei.zhang@qq.com

@time: 2018/9/11 10:30
@python version: 

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 数据预处理
dataset = pd.read_csv("../datasets/studentscores.csv")
X = dataset.iloc[:, :1].values
Y = dataset.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# 使用简单线性模型来训练
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# 预测测试集数值
Y_pred = regressor.predict(X_test)

# 训练集
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()

# 测试集
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.show()
