import numpy as np

npts = 50
x = np.linspace(0, 1, npts)
x = np.asarray(x).reshape(-1, 1)
Amat = []
degree = 50
for i in range(degree + 1):
    ithCol = x**i
    Amat.append(ithCol)
Amat = np.asarray(Amat).reshape(-1, 50).T

[u, s, v] = np.linalg.svd(Amat)

def smat_reduced(shape, diag, n):
    smat = np.zeros(shape, dtype = complex)
    drop = len(diag) - n
    if drop > 0:
        diag = np.concatenate((diag[ : -drop], np.zeros(drop)))
    smat[:len(diag), :len(diag)] = np.diag(diag)
    return smat

def svd_to_mat(u, s, v, n):
    smat= smat_reduced((len(u), len(v)), s, n)
    return u.dot(smat).dot(v)

def svd_reconstruction_error(Amat, u, s, v, n):
    error = np.square(Amat - svd_to_mat(u,s,v,n)).sum()
    return error
n = 50
error = svd_reconstruction_error(Amat, u, s, v, n)