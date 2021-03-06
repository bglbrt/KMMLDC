#!/usr/bin/env python

# file management libraries
import importlib

# numerical libraries
import numpy as np
import scipy.sparse.linalg as ssl
import scipy.spatial.distance as ssd

# dependencies
from kernels import *

# Kernel Fisher Discriminant Analysis classifier
class KFDA():
    '''
    Kernel Fisher Discriminant Analysis classifier.
    '''

    def __init__(self):
        pass

    def fit(self, Xtr, Ytr, kernel, gamma, n_components, kernel_kwargs):
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
            - n_components: int
                number of components for KFDA
            - kernel_kwargs: dict
                arguments to pass to kernel class
        '''

        # set data
        self.Xtr = Xtr
        self.Ytr = Ytr

        # set number of classes and number of components
        self.n_classes = len(set(self.Ytr))
        self.n_components = n_components

        # one-hot encode Y vector
        self.Ytr_ohe = np.eye(self.n_classes)[self.Ytr]

        # set parameters
        self.kernel_class = getattr(importlib.import_module("kernels"), kernel)
        self.kernel = self.kernel_class(**kernel_kwargs)
        self.gamma = gamma

        # compute Gram matrix
        self.K = self.kernel.compute(self.Xtr, self.Xtr)

        # compute class means
        self.classes_means = np.dot(self.K, self.Ytr_ohe / np.sum(self.Ytr_ohe, axis=0)).T

        # compute N
        self.N = np.dot(self.K, self.K-self.classes_means[self.Ytr]) + self.gamma * np.eye(self.Xtr.shape[0])

        # compute M
        self.classes_means_centered = self.classes_means - np.mean(self.K, axis=1)
        self.M = np.dot(self.classes_means_centered.T, self.classes_means_centered)

        # compute weights
        self.alpha = ssl.eigsh(self.M, self.n_components, self.N)[1]

        # compute centroids
        self.centroids = np.dot(self.classes_means, self.alpha)

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
        pte = np.dot(self.kernel.compute(Xte, self.Xtr), self.alpha)

        # find closest centroids
        Yte = np.argmin(ssd.cdist(pte, self.centroids), axis=1)

        # return output
        return Yte
