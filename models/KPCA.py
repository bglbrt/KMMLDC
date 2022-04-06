#!/usr/bin/env python

# file management libraries
import importlib

# numerical libraries
import numpy as np
from more_itertools import nth_product
import scipy.sparse.linalg as ssl
import scipy.spatial.distance as ssd
from sympy import nth_power_roots_poly

# dependencies
from kernels import *

# Kernel Fisher Discriminant Analysis classifier
class KPCA():
    '''
    Kernel Principal Component Analysis classifier.
    '''

    def __init__(self):
        pass

    def centering_K(self):
        #N = self.K.shape[0]
        #I = np.eye(N)
        #U = np.ones(self.K.shape)/N

        ### A changer ###
        #self.center_vec = self.K.mean(axis=1) - self.K.mean()
        n_samples = self.K.shape[0]
        self.K_fit_rows_ = np.sum(self.K, axis=0) / n_samples
        self.K_fit_all_ = self.K_fit_rows_.sum() / n_samples
        #self.K = (I - U) @ self.K @ (I - U)

    def fit(self, Xtr, Ytr, kernel, n_components, decomp_kernel_kwargs):
        '''
        Fitting function.

        Arguments:
            - Xtr: np.array
                X train data
            - Ytr: np.array
                Y train data
            - kernel: str
                name of kernel
            - gamma: float
                Kernel Fisher Discriminant Analysis regularization parameter
        '''

        # set data
        self.Xtr = Xtr
        self.Ytr = Ytr
        N = len(self.Ytr)

        # set number of classes and number of components
        self.n_classes = len(set(self.Ytr))
        self.n_components = n_components

        # set parameters
        self.kernel_class = getattr(importlib.import_module("kernels"), kernel)
        self.kernel = self.kernel_class(**decomp_kernel_kwargs)

        # compute Gram matrix
        self.K = self.kernel.compute(self.Xtr, self.Xtr)
        self.centering_K()

        # compute eigenvectors
        eig_val, eig_vec = np.linalg.eigh(self.K)
        idx = eig_val.argsort()[::-1]
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:,idx]
        self.alpha = eig_vec[:,eig_val > 0] / np.sqrt(eig_val[eig_val > 0])

        # project on n_components
        self.alpha = self.alpha[:,:n_components]

        n, m = self.alpha.shape
        variance = np.trace(self.K) / n
        projection = np.dot(self.K, self.alpha)
        percentage_explained = np.linalg.norm(projection[:, :n_components]) ** 2 / n / variance * 100.0
        print('Variance explained:', percentage_explained)


    def predict(self, Xte):
        '''
        Predict function.

        Arguments:
            - Xte: np.array
                input data

        Returns:
            - Yte: np.array
                prediction
        '''

        # compute transform of Xte in space of KFDA
        K = self.kernel.compute(Xte, self.Xtr)
        K_pred_cols = (np.sum(K, axis=1) / self.K_fit_rows_.shape[0])[:, np.newaxis]

        K -= self.K_fit_rows_
        K -= K_pred_cols
        K += self.K_fit_all_
        #K = K - K.mean(axis=1)[:,np.newaxis] - self.center_vec

        # find closest centroids
        Yte = K @ self.alpha

        # return output
        return Yte


if __name__ == '__main__':
    from sklearn.datasets import make_circles
    import matplotlib.pyplot as plt
    from sklearn.decomposition import KernelPCA

    f, axarr = plt.subplots(2, 2, sharex=True)

    X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
    X[:,1] += 0.5
    axarr[0, 0].scatter(X[y==0, 0], X[y==0, 1], color='red')
    axarr[0, 0].scatter(X[y==1, 0], X[y==1, 1], color='blue')

    kpca = KPCA()
    kernel_kwargs = {}
    kpca.fit(X, y, 'Linear', 1, kernel_kwargs)
    Xproj = kpca.predict(X)
    axarr[0, 1].scatter(Xproj[y==0, 0], np.zeros(500), color='red')
    axarr[0, 1].scatter(Xproj[y==1, 0], np.zeros(500), color='blue')

    # decrease sigma to improve separation
    #kpca = KPCA(RBF(0.686))
    kernel_kwargs = {'sigma': 0.25}
    kpca.fit(X, y, 'RBF', 2, kernel_kwargs)
    print(kpca.alpha.shape[1])
    Xproj = kpca.predict(X)
    axarr[1, 0].scatter(Xproj[y==0, 0], np.zeros(500), color='red')
    axarr[1, 0].scatter(Xproj[y==1, 0], np.zeros(500), color='blue')

    axarr[1, 1].scatter(Xproj[y==0, 0], Xproj[y==0, 1], color='red')
    axarr[1, 1].scatter(Xproj[y==1, 0], Xproj[y==1, 1], color='blue')

    plt.show()

    f2, axarr2 = plt.subplots(2, 2, sharex=True)

    X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
    X[:,1] += 0.5
    axarr2[0, 0].scatter(X[y==0, 0], X[y==0, 1], color='red')
    axarr2[0, 0].scatter(X[y==1, 0], X[y==1, 1], color='blue')

    kpca = KernelPCA(n_components=1, kernel='linear')
    kpca.fit(X, y)
    Xproj = kpca.transform(X)
    axarr2[0, 1].scatter(Xproj[y==0, 0], np.zeros(500), color='red')
    axarr2[0, 1].scatter(Xproj[y==1, 0], np.zeros(500), color='blue')

    # decrease sigma to improve separation
    #kpca = KPCA(RBF(0.686))
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
    kpca.fit(X, y)
    Xproj = kpca.transform(X)
    axarr2[1, 0].scatter(Xproj[y==0, 0], np.zeros(500), color='red')
    axarr2[1, 0].scatter(Xproj[y==1, 0], np.zeros(500), color='blue')

    axarr2[1, 1].scatter(Xproj[y==0, 0], Xproj[y==0, 1], color='red')
    axarr2[1, 1].scatter(Xproj[y==1, 0], Xproj[y==1, 1], color='blue')

    plt.show()
