import numpy as np



def reg_gradientdescent(winit, Amat, y_true, rate, Iter, l1, l2):
    w = winit
    w_history = []
    for i in range(Iter):
        grad = -(2/n) * Amat.T.dot(y_true - Amat.dot(w)) + l1 * np.sign(w) + 2 * l2 * w
        w = w - rate * grad
        w_history.append(w)
    return np.asarray(w_history)

