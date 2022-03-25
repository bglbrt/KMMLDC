#!/usr/bin/env python

# os libraries
import os

# numerical and computer vision libraries
import numpy as np

# linear kernel
class Linear:
    '''
    Linear kernel.
    '''

    def __init__(self):
        pass

    def compute(self, X1, X2):
        '''
        Kernel computation function.

        Arguments:
            - X1: np.array
                N x d matrix
            - X2: np.array
                M x d matrix

        Returns:
            - G: np.array
                N x M Gram matrix
        '''

        # compute Gram matrix
        G = np.tensordot(X, Y, axes=(1, 1))

        # return G
        return G

# RBF kernel
class RBF():
    '''
    Radial-Basis Function kernel.

    Arguments:
        - sigma: float
            bandwith parameter for the RBF kernel
    '''

    def __init__(self, sigma=1.):

        # set bandwith parameter
        self.sigma = sigma

    def compute(self, X1, X2):
        '''
        Kernel computation function.

        Arguments:
            - X1: np.array
                N x d matrix
            - X2: np.array
                M x d matrix

        Returns:
            - G: np.array
        '''

        # compute norm of observations in X and Y
        X1N = np.square(np.linalg.norm(X1, axis=1))
        X2N = np.square(np.linalg.norm(X2, axis=1))

        # compute ||xi-yi||^2 = ||xi||^2 + ||yi^2|| - 2 <xi, yi>
        O = np.add.outer(X1N, X2N) - 2 * np.tensordot(X1, X2, axes=(1, 1))

        # compute Gram matrix
        G = np.exp(-(1/(2 * (self.sigma**2))) * O)

        # return G
        return G
