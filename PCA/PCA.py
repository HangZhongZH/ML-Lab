import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as spio

rng = np.random.RandomState(1)
xMean = np.array([2, 4, -8])
xCov = np.array([[8.5, -3.5, 2.1], [-3.5, 12, 4.5], [2.1, 4.5, 3.5]])
npts = 10000
x = np.random.multivariate_normal(xMean, xCov, npts)
fig = plt.figure()
ax0 = fig.add_subplot(121, projection = '3d')
ax0.scatter(x[:, 0], x[:, 1],x[:, 2])
plt.show()

xFeartureMean = np.mean(x, axis = 0)
xZeroMean = x - xFeartureMean
[u, s, v] = np.linalg.svd(xZeroMean)
v = v.T



#Apply PCA on a small face dataset
data = spio.loadmat('/Users/zhonghang/PycharmProjects/MLLab/Imgs.mat')
images = data['Imgs']
imat = images.T

fig, ax = plt.subplots(12, 7, figsize = (17, 18))
for i in range(12):
    for j in range(7):
        ax[i][j].imshow(imat[i * 7 + j, :].reshape(57, 72), cmap = 'gray')
        
imMean = np.mean(imat, axis=0)
imBias = imat - imMean
[u, s, v] = np.linalg.svd(imBias)
eigvlPercent = s**2 / np.sum(s**2)
eigvlCumPercent = np.zeros(s.shape)
for i in range(len(eigvlCumPercent)):
    eigvlCumPercent[i] = np.sum(eigvlPercent[: i + 1])
print(eigvlCumPercent)
#Eigvenfaces
fig, ax = plt.subplots(2, 4)
ax[0][3].imshow(imMean.reshape(57, 72), cmap = 'gray')
ax[0][3].set_xlabel('mean')
for i in range(2):
    for j in range(3):
        ax[i][j].imshow(v.T[i * 3 + 1].reshape(57, 72), cmap = 'gray')
plt.show()

fig, ax = plt.subplots(5, 7)
for i in range(5):
    for j in range(7):
        ith = i * 5 + j
        ax[i][j].imshow(imat[ith, :].reshape(57, 72), cmap = 'gray')
plt.show()
