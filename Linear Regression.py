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