# Use linear regression to fit non-linear models
def f2(x):
    y = (x/3)**2 * np.exp(-(x/6)**2)
    return y

x = np.linspace(0, 20, 50)
y2 = f2(x) + np.random.normal(0, 1, 50)
x = x.reshape(-1, 1)
y2 = y2.reshape(-1, 1)
x_train, y_train