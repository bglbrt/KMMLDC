#!/usr/bin/env python

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

    def fit(self, Xtr, Ytr, kernel='RBF', gamma=0.1):
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
                Ridge Regression classifier regularization parameter
        '''

        # set data
        self.Xtr = Xtr
        self.Ytr = Ytr

        # set parameters
        self.kernel = kernels[kernel]
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
        Yte = np.dot(self.alpha, self.kernel.compute(Xte, self.Xtr).T)

        # set output to closest integer
        Yte = np.round(np.minimum(np.maximum(Yte, 0), 9))

        # return output
        return Yte

# dictionary of classifiers
classifiers = {'KRR': KRR()}
