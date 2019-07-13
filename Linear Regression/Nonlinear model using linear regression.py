# Use linear regression to fit non-linear models
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def f2(x):
    y = (x/3)**2 * np.exp(-(x/6)**2)
    return y

x = np.linspace(0, 20, 50)
y2 = f2(x) + np.random.normal(0, 0.1, 50)
x = x.reshape(-1, 1)
y2 = y2.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y2, test_size = 0.33)

plt.figure()
plt.scatter(x_train, y_train, c = 'b', label = 'train')
plt.scatter(x_test, y_test, c = 'k', label = 'test')
plt.legend()
plt.show()

def polyfn(x, w):
    n = len(w)
    result = 0
    for i in range(n):
        result += w[i] *np.power(x, n - i - 1)
    return result

def designmatpoly(x):
    third_col = np.ones(x.shape)
    second_col= x
    first_col = polyfn(x, [1, 0, 0])
    Amat = np.concatenate((first_col, second_col, third_col), axis = 1)
    return Amat

A_trainb= designmatpoly(x_train)
A_test = designmatpoly(x_test)