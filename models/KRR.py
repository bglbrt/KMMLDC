#!/usr/bin/env python

# file management libraries
import importlib

# numerical libraries
import numpy as np

# dependencies
from kernels import *

# Kernel Ridge regression classifier
class KRR():
    '''
    Kernel Ridge regression classifier.
    '''

    def __init__(self):
        pass

    def fit(self, Xtr, Ytr, kernel, gamma, kernel_kwargs):
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
                Kernel Ridge Regression classifier regularization parameter
        '''

        # set data
        self.Xtr = Xtr
        self.Ytr = Ytr

        # set parameters
        self.kernel_class = getattr(importlib.import_module("kernels"), kernel)
        self.kernel = self.kernel_class(**kernel_kwargs)
        self.gamma = gamma

        # compute Gram matrix
        self.K = self.kernel.compute(self.Xtr, self.Xtr)

        # compute alpha
        self.alpha = np.dot(np.linalg.inv(self.K + self.gamma * np.eye(self.Xtr.shape[0])), self.Ytr)

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

        # compute output
        Yte = np.dot(self.kernel.compute(Xte, self.Xtr), self.alpha)

        # set output to closest integer
        Yte = np.round(np.minimum(np.maximum(Yte, 0), 9))

        # return output
        return Yte
