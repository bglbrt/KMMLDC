#!/usr/bin/env python

# numerical libraries
import numpy as np

# linear kernel
class Linear():
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
        G = np.tensordot(X1, X2, axes=(1, 1))

        # return G
        return G

# polynomial kernel
class Polynomial():
    '''
    Polynomial kernel.

    Arguments:
        - a: float
            - affine parameter for polynomial kernel
        - c: float
            - bias parameter for polynomial kernel
        - d: int
            - power parameter for polynomial kernel
    '''

    def __init__(self, a=1.0, c=1.0, d=1):

        # set polynomial parameters
        self.a = a
        self.c = c
        self.d = d

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
        G = np.power(a * np.tensordot(X, Y, axes=(1, 1)) + c, d)

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

    def __init__(self, sigma=5.):

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

# Exponential kernel
class Exponential():
    '''
    Exponential kernel.

    Arguments:
        - sigma: float
            bandwith parameter for the Exponential kernel
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
        O[O < 0] = 0

        # compute Gram matrix
        G = np.exp(-(1/(2 * (self.sigma**2))) * np.sqrt(O))

        # return G
        return G

# Laplacian kernel
class Laplacian():
    '''
    Laplacian kernel.

    Arguments:
        - sigma: float
            bandwith parameter for the Laplacian kernel
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
        O[O < 0] = 0

        # compute Gram matrix
        G = np.exp(-(1/self.sigma) * np.sqrt(O))

        # return G
        return G

# TanH kernel
class TanH():
    '''
    Hyperbolic tangent kernel.

    Arguments:
        - a: float
            - affine parameter for hyperbolic tangent kernel
        - c: float
            - bias parameter for hyperbolic tangent kernel
    '''

    def __init__(self, a=1.0, c=1.0):

        # set polynomial parameters
        self.a = a
        self.c = c

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
        G = np.tanh(a * np.tensordot(X, Y, axes=(1, 1)) + c)

        # return G
        return G

# Multiquadratic kernel
class Multiquadratic():
    '''
    Multiquadratic kernel.

    Arguments:
        - c: float
            bias parameter for multiquadratic kernel
    '''

    def __init__(self, c=1.):

        # set bandwith parameter
        self.c = c

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
        O[O < 0] = 0

        # compute Gram matrix
        G = np.sqrt(O + self.c**2)

        # return G
        return G

# Multiquadratic kernel
class InverseMultiquadratic():
    '''
    InverseMultiquadratic kernel.

    Arguments:
        - c: float
            bias parameter for inverse multiquadratic kernel
    '''

    def __init__(self, c=1.):

        # set bandwith parameter
        self.c = c

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
        O[O < 0] = 0

        # compute Gram matrix
        G = np.reciprocal(np.sqrt(O + self.c**2))

        # return G
        return G

# Histogram Intersection
class HistogramIntersection():
    '''
    HistogramIntersection kernel.
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
        '''

        # initialise Gram matrix
        G = np.zeros((X1.shape[0], X2.shape[0]))

        # fill Gram matrix
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                G[i, j] = np.sum(np.minimum(X1[i], X2[j]))

        # return G
        return G

# Log kernel
class Log():
    '''
    Log kernel.

    Arguments:
        - d: int
            power parameter for Log kernel
    '''

    def __init__(self, d=1):

        # set power parameter
        self.d = d

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

        # compute O = ||xi-yi||^2 = ||xi||^2 + ||yi^2|| - 2 <xi, yi>
        O = np.add.outer(X1N, X2N) - 2 * np.tensordot(X1, X2, axes=(1, 1))
        O[O < 0] = 0

        # compute A = sqrt(O)
        A = np.sqrt(O)

        # compute Gram matrix G = log(||xi-yi||^d + 1)
        G = -np.log(np.power(A, self.d) + 1)

        # return G
        return G

# dictionary of kernels
kernels = {'Linear': Linear(),
           'Polynomial': Polynomial(),
           'RBF': RBF(),
           'Exponential': Exponential(),
           'Laplacian': Laplacian(),
           'TanH': TanH(),
           'Multiquadratic': Multiquadratic(),
           'InverseMultiquadratic': InverseMultiquadratic(),
           'HistogramIntersection': HistogramIntersection(),
           'Log': Log()}
