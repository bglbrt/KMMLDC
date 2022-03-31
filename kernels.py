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

    def __init__(self, a, c, d):

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
        G = np.power(self.a * np.tensordot(X1, X2, axes=(1, 1)) + self.c, self.d)

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

    def __init__(self, sigma):

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

    def __init__(self, sigma):

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

        # correct for numerical errors
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

    def __init__(self, sigma):

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
                Gram matrix
        '''

        # compute norm of observations in X and Y
        X1N = np.square(np.linalg.norm(X1, axis=1))
        X2N = np.square(np.linalg.norm(X2, axis=1))

        # compute ||xi-yi||^2 = ||xi||^2 + ||yi^2|| - 2 <xi, yi>
        O = np.add.outer(X1N, X2N) - 2 * np.tensordot(X1, X2, axes=(1, 1))

        # correct for numerical errors
        O[O < 0] = 0

        # compute Gram matrix
        G = np.exp(-(1/self.sigma) * np.sqrt(O))

        # return G
        return G

# ANOVA kernel
class ANOVA():
    '''
    ANOVA kernel.

    Arguments:
        - sigma: float
            bandwith parameter for the ANOVA kernel
        - d: int
            power parameter for the ANOVA kernel
    '''

    def __init__(self, sigma, d):

        # set parameters for ANOVA kernel
        self.sigma = sigma
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

        # get number of features
        n_feat = X1.shape[1]

        # initialise Gram matrix
        G = np.zeros((X1.shape[0], X2.shape[0]))

        # populate Gram matrix with all features
        for k in range(n_feat):
            G += np.power(np.exp(-self.sigma * np.subtract.outer(X1[:,k], X2[:,k])), self.d)

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

    def __init__(self, a, c):

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

# Rational Quadratic kernel
class RationalQuadratic():
    '''
    Rational Quadratic kernel.

    Arguments:
        - c: float
            regularization parameter for the Rational Quadratic kernel
    '''

    def __init__(self, c):

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
                Gram matrix
        '''

        # compute norm of observations in X and Y
        X1N = np.square(np.linalg.norm(X1, axis=1))
        X2N = np.square(np.linalg.norm(X2, axis=1))

        # compute ||xi-yi||^2 = ||xi||^2 + ||yi^2|| - 2 <xi, yi>
        O = np.add.outer(X1N, X2N) - 2 * np.tensordot(X1, X2, axes=(1, 1))

        # correct for numerical errors
        O[O < 0] = 0

        # compute Gram matrix
        G = - np.divide(O, O+c) + 1

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

    def __init__(self, c):

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

        # correct for numerical errors
        O[O < 0] = 0

        # compute Gram matrix
        G = np.sqrt(O + self.c**2)

        # return G
        return G

# Inverse Multiquadratic kernel
class InverseMultiquadratic():
    '''
    InverseMultiquadratic kernel.

    Arguments:
        - c: float
            bias parameter for Inverse Multiquadratic kernel
    '''

    def __init__(self, c):

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

        # correct for numerical errors
        O[O < 0] = 0

        # compute Gram matrix
        G = np.reciprocal(np.sqrt(O + self.c**2))

        # return G
        return G

# Wave kernel
class Wave():
    '''
    Wave kernel.

    Arguments:
        - c: float
            regularization parameter for Wave kernel
        - theta: float
            angle parameter for Wave kernel
    '''

    def __init__(self, c, theta):

        # set parameters for Wave kernel
        self.c = c
        self.theta = theta

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

        # correct for numerical errors
        O[O < 0] = 0

        # compute A = ||xi-yi|| from O
        A = np.sqrt(O)

        # compute Gram matrix
        G = self.theta * np.reciprocal(A + self.c) * np.sin((1/self.theta) * A)

        # return G
        return G

# Wave kernel
class Power():
    '''
    Power kernel.

    Arguments:
        - d: int
            power parameter for Power kernel
    '''

    def __init__(self, d):

        # set power parameter for Power kernel
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

        # compute ||xi-yi||^2 = ||xi||^2 + ||yi^2|| - 2 <xi, yi>
        O = np.add.outer(X1N, X2N) - 2 * np.tensordot(X1, X2, axes=(1, 1))

        # correct for numerical errors
        O[O < 0] = 0

        # compute A = ||xi-yi|| from O
        A = np.sqrt(O)

        # compute Gram matrix
        G = - np.power(A, self.d)

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

    def __init__(self, d):

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

        # correct for numerical errors
        O[O < 0] = 0

        # compute A = sqrt(O)
        A = np.sqrt(O)

        # compute Gram matrix G = log(||xi-yi||^d + 1)
        G = -np.log(np.power(A, self.d) + 1)

        # return G
        return G

# Cauchy kernel
class Cauchy():
    '''
    Cauchy kernel.

    Arguments:
        - sigma: float
            variance parameter for Cauchy kernel
    '''

    def __init__(self, sigma):

        # set variance parameter for Cauchy kernel
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

        # compute O = ||xi-yi||^2 = ||xi||^2 + ||yi^2|| - 2 <xi, yi>
        O = np.add.outer(X1N, X2N) - 2 * np.tensordot(X1, X2, axes=(1, 1))

        # correct for numerical errors
        O[O < 0] = 0

        # compute A = sqrt(O)
        G = np.reciprocal(1 + (1/self.sigma**2) * np.sqrt(O))

        # return G
        return G

# Chi Square kernel
class ChiSquare():
    '''
    Chi Square kernel.
    '''

    def __init__(self, sigma):

        # set variance parameter for Chi Square kernel
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

        # get number of features
        n_feat = X1.shape[1]

        # initialise Gram matrix
        G = np.zeros((X1.shape[0], X2.shape[0]))

        # populate Gram matrix with all features
        for k in range(n_feat):
            G += np.divide(np.square(np.subtract.outer(X1[:,k], X2[:,k])), np.add.outer(X1[:,k], X2[:,k]))

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

        # get number of features
        n_feat = X1.shape[1]

        # fill Gram matrix
        for m in range(n_feat):
            #print(m)
            G += np.minimum(X1[:, m].reshape(-1, 1), X2[:, m].reshape(-1, 1).T)

        # return G
        return G
