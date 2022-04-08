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

# Kernel Principal Component Analysis
class KPCA():
    '''
    Kernel Principal Component Analysis.
    '''

    def __init__(self):
        pass

    def fit_centering(self, K):
        '''
        Fit centering components from Gram matrix.

        Arguments:
            - K: np.array
                Gram matrix
        '''

        # compute centers of columns and matrix
        self.K_center_cols = K.sum(axis=0) / self.N
        self.K_center_all = self.K_center_cols.sum() / self.N

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
            - n_components: int
                number of principal components
        '''

        # set data
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.N = len(self.Ytr)

        # set number of classes and number of components
        self.n_classes = len(set(self.Ytr))
        self.n_components = n_components

        # set parameters
        self.kernel_class = getattr(importlib.import_module("kernels"), kernel)
        self.kernel = self.kernel_class(**decomp_kernel_kwargs)

        # compute and fit centering of Gram matrix
        K = self.kernel.compute(self.Xtr, self.Xtr)
        self.fit_centering(K)

        # compute eigenvectors and sort in decreasing order of eigenvalues
        eig_val, eig_vec = np.linalg.eigh(K)
        idx = eig_val.argsort()[::-1]
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:,idx]
        self.alpha = eig_vec[:,eig_val > 0] / np.sqrt(eig_val[eig_val > 0])

        # project on n_components
        self.alpha = self.alpha[:,:n_components]

        # print variance explained
        var = np.trace(K) / self.N
        var_explained = np.linalg.norm((K @ self.alpha)) ** 2 / (self.N * var)
        print('Variance explained:', np.round(var_explained,3))


    def transform(self, Xte):
        '''
        Transform function.

        Arguments:
            - Xte: np.array of shape (N,d)
                input data

        Returns:
            - Xte_new: np.array of shape (N,n_components)
                projected data
        '''

        # center kernel matrix between Xte and Xtr (training data)
        K = self.kernel.compute(Xte, self.Xtr)
        K_center_vec = (K.sum(axis=1) / self.K_center_cols.shape[0])[:, np.newaxis]
        K -= self.K_center_cols
        K -= K_center_vec
        K += self.K_center_all

        # projection on n_components (principal components)
        Xte_new = K @ self.alpha

        # return output
        return Xte_new
