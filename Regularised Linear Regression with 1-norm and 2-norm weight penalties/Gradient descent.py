import numpy as np
import matplotlib.pyplot as plt

def gradientdescent(Amat, y_true, w_init, rate, Iter):
    n, p = Amat.shape
    w = w_init
    w_history = []
    w_history.append(w)
    loss_history = []
    for i in range(Iter):
        loss = np.square(y_true - Amat.dot(w)).mean()
        loss_history.append(loss)
        grad = -(2/n) * Amat.T.dot(y_true - Amat.dot(w))
        w = w - rate * grad
        w_history.append(w)
    return w_history, loss_history
    
def SGDescent(Amat, y_true, w_init, tn, td, epochs):
    n, p = Amat.shape
    w = w_init
    w_history = []
    loss_history = []
    w_all = []
    w_all.append(w)
    for i in range(epochs):
        loss = np.square(y_true - Amat.dot(w)).mean()
        w_history.append(w)
        loss_history.append(loss)
        for j in range(n):
            idx = np.random.randint(n)
            xi = Amat[idx: idx + 1, :]
            yi = y_true[idx: idx + 1]
            grad = -(2/n) * xi.T.dot(yi - xi.dot(w))
            rate = tn / (td + (n * i + j))
            w = w - rate * grad
            w_all.append(w)
    return w_history, loss_history, w_all


def f(x):
    y = 0.5 * x**5 / (.05 + x**4)
    return y

npts = 5
np.random.seed(0)
x = np.linspace(0, 1, npts)
x = x.reshape(-1, 1)
y_true = f(x) + 0.03 * np.random.normal(0, 1, [npts, 1])
y_true = y_true.reshape(-1, 1)
firstCol = np.square(x)
secondCol = x
thirdCol = np.ones([npts, 1])
Amat = np.concatenate((firstCol, secondCol, thirdCol), axis = 1)
xTest = np.linspace(0, 1, 50)
xTest = xTest.reshape(-1, 1)

rates = [.0001, .01, .05, .1, .5]
fig, ax = plt.subplots(1, 2)
ax[0].scatter(x, y_true)
w_init = np.random.random([Amat.shape[1], 1])
for idx, item in enumerate(rates):
    _, loss_history = gradientdescent(Amat, y_true, w_init, item, 20)
    ax[1].plot(loss_history, label = item)
ax[1].legend()
plt.show()


#Use LSE(pseudo inverse)
#w = (AA.T)**(-1)Ay
npts = 8
np.random.seed(0)
x = np.linspace(0, 1, npts)
x = x.reshape(-1, 1)
y0 = f(x) + 0.03 * np.random.normal(0, 1, x.shape)
firstCol = np.square(x)
secondCol = x
#thirdCol = np.ones(x.shape)
Amat = np. concatenate((firstCol, secondCol), axis = 1)
wBest = np.linalg.pinv(Amat.T.dot(Amat)).dot(Amat.T).dot(y0)

w0 = np.random.random([Amat.shape[1], 1])
epochs = 15
tnlist = [0.01, .1, 1., 10.]
td = 15
wlist = []
fig, ax = plt.subplots(1, 3)
for i in tnlist:
    w_history, loss_history, w_all = SGDescent(Amat, y0, w0, i, td, epochs)
    wlist.append(w_history[-1])
    ax[0].plot(loss_history, label = i)
ax[0].legend()

for idx, item in enumerate(wlist):
    ax[1].plot(x, Amat.dot(item), label = tnlist[idx])
ax[1].plot(x, Amat.dot(wBest), label = 'LSE')
ax[1].scatter(x, y0, label = 'True data')
ax[1].legend()

npts = 10
np.random.seed(0)
x_test = np.linspace(0, 2, npts)
x_test = x_test.reshape(-1, 1)
y_true_test = f(x_test) + 0.05 * np.random.normal(0, 1, x_test.shape)
firstCol = np.square(x_test)
secondCol = x_test
#thirdCol = np.ones(x_test.shape)
Amat = np.concatenate((firstCol, secondCol), axis = 1)
y_predict_LSE = Amat.dot(wBest)
ax[2].scatter(x_test, y_true_test, label = 'True data')
ax[2].plot(x_test, y_predict_LSE, label = 'LSE')
for idx, item in enumerate(wlist):
    ax[2].plot(x_test, Amat.dot(item), label = tnlist[idx])
ax[2].legend()
plt.show()


