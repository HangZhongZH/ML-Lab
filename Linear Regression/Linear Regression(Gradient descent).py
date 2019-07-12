import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy as sp
from sklearn import datasets


def f1(x, m, c):
    y = m * x + c
    return y

xmin, xmax, pointsnumber = -4, 10, 50
X = np.linspace(xmin, xmax, pointsnumber)

plt.figure('In2')
y0 = f1(X, -3., 9.) + np.random.normal(0., 4., X.shape)
plt.scatter(X, y0, c = 'b', marker= 'x')
plt.show()

def designmat1(X):
    X = X.reshape(X.shape[0], 1)
    ConstCol = np.ones([X.shape[0], 1])
    Xmat = np.concatenate((X, ConstCol), axis = 1)
    return Xmat



def gradsqloss(Xmat, y_true, w):
    n = Xmat.shape[0]
    y_true = y_true.reshape(y_true.shape[0], 1)
    grad = -(2/n) * (np.dot((y_true - Xmat.dot(w)).T, Xmat)).T
    return grad

def graddescent(Xmat, y_true, w_init, rate, IterNum):
    w_all = []
    w = w_init
    loss = []
    for iter in range(IterNum):
        grad = gradsqloss(Xmat, y_true, w)
        w = w - rate * grad
        w_all.append(w)
        y_predict = Xmat.dot(w)
        loss_temp = np.square(y_predict - y_true.reshape(-1, 1)).mean()
        loss.append(loss_temp)
    return w_all, loss

Xmat = designmat1(X)
col = Xmat.shape[1]
w_init = np.random.randn(col, 1)
y_true = y0
rate = [.001, .005, .01, .02]
IterNum = 100


for idx, item in enumerate(rate):
    w_all, loss = graddescent(Xmat, y_true, w_init, item, IterNum)
    xaxis = np.linspace(1, IterNum + 1, 100)
    plt.subplot(211)
    plt.plot(xaxis, loss)
    w_all = np.array(w_all[-1]).reshape(2, 1)
    y1 = X * w_all[0] + w_all[1]
    plt.subplot(212)
    plt.plot(X, y1)

plt.scatter(X, y0, c = 'b', marker= 'x')
plt.show()










#Using scikit-learn to get the solution
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y0, test_size = 0.33)
plt.figure()
plt.scatter(x_train, y_train, c = 'b', label = 'train')
plt.scatter(x_test, y_test, c = 'k', marker = '+', label = 'test')
plt.legend()
plt.show()

from sklearn import linear_model
regr = linear_model.LinearRegression()
x_train, y_train, x_test, y_test = x_train.reshape(-1, 1), y_train.reshape(-1, 1),         x_test.reshape(-1, 1), y_test.reshape(-1, 1)
regr.fit(x_train, y_train)
slope = regr.coef_.reshape(-1)
intercept = regr.intercept_.reshape(-1)
y_predict = (slope * x_train + intercept).reshape(-1)
y_predict2 = regr.predict(x_train).reshape(-1)
print((y_predict == y_predict2).all())


y_test_predict2 = regr.predict(x_test).reshape(-1)
slope = regr.coef_.reshape(-1)
intercept = regr.intercept_.reshape(-1)
y_test_predict = slope * x_test + intercept
y_test_predict = y_test_predict.reshape(-1)
print((y_test_predict == y_test_predict2).all())

plt.figure()
plt.scatter(x_train, y_train, c = 'b', label = 'true')
plt.scatter(x_train, y_predict, c = 'r', label = 'predict')
plt.legend()
plt.show()

plt.figure()
plt.scatter(x_test, y_test, c = 'k', label = 'true')
plt.scatter(x_test, y_test_predict, c= 'b', label = 'predict')
plt.legend()
plt.show()














