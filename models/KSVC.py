#!/usr/bin/env python



# file management libraries
import importlib
from tqdm import tqdm
import random

# numerical libraries
import numpy as np
from scipy import optimize

# dependencies
from kernels import *

# Support Vector Classifier
class SVC():
    '''
    Support Vector Classifier.
    '''

    def __init__(self, epsilon = 1e-3):
        self.alpha = None
        self.epsilon = epsilon

    def fit(self, Xtr, Ytr, kernel, C, kernel_kwargs):
        '''
        Fitting function.

        Arguments:
            - Xtr: np.array
                X train data
            - Ytr: np.array
                Y train data
            - kernel: str
                name of kernel
            - C: float
                some parameter
            - gamma: float
                Kernel Support Vector regularization parameter
        '''

        # set data
        self.Xtr = Xtr

        # set parameters
        self.C = C
        self.kernel_class = getattr(importlib.import_module("kernels"), kernel)
        self.kernel = self.kernel_class(**kernel_kwargs)
        #self.gamma = gamma

        # compute Gram matrix
        K = self.kernel.compute(self.Xtr, self.Xtr)

        # define useful variables
        N = len(Ytr)
        diag_y = np.diag(Ytr)

        # Lagrange dual problem
        def loss(alpha):

            # Dual Loss
            return 1/2*np.linalg.multi_dot([alpha.T, diag_y, K, diag_y, alpha])\
                    - alpha.T.dot(np.ones(N))

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):

            # Partial derivative of the dual loss wrt alpha
            return np.linalg.multi_dot([diag_y, K, diag_y, alpha]) - np.ones(N)


        # Equality constraints
        fun_eq = lambda alpha: -alpha.T.dot(Ytr)
        jac_eq = lambda alpha: -Ytr

        # Inequality constraints
        fun_ineq = lambda alpha: self.C*np.ones(N) - alpha
        jac_ineq = lambda alpha: - np.eye(N)
        fun_ineq_pos = lambda alpha: alpha
        jac_ineq_pos = lambda alpha: np.eye(N)

        constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq', 'fun': fun_ineq , 'jac': jac_ineq},
                       {'type': 'ineq', 'fun': fun_ineq_pos, 'jac': jac_ineq_pos})

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N),
                                   method='SLSQP',
                                   jac=lambda alpha: grad_loss(alpha),
                                   constraints=constraints)
        self.alpha = optRes.x

        # Select indices of support vectors
        supportIndices = np.where((self.alpha > self.epsilon)\
                                  & (C*np.ones(N) - self.alpha > self.epsilon))

        # Compute diag(y).dot(alpha) to avoid computing it at inference
        self.alpha = diag_y.dot(self.alpha)

        # Offset of the classifier
        self.b = np.mean(Ytr[supportIndices] - self.alpha.T.dot(K)[supportIndices])

    ### Implementation of the separting function $f$
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return self.alpha.T.dot(self.kernel.compute(self.Xtr,x))

    def predict(self, Xte):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(Xte)
        return 2 * (d+self.b> 0) - 1


class OvOKSVC():
    def __init__(self):
        self.clfs = []

    def fit(self, Xtr, Ytr, kernel, C, kernel_kwargs):
        self.n_classes = len(np.unique(Ytr))
        comb = [(pos,neg) for pos in range(self.n_classes) for neg in range(pos+1, self.n_classes)]
        for pos, neg in tqdm(comb):
            #print('Training:', pos, 'VS', neg)
            X, Y = self.make_data(Xtr, Ytr, pos, neg)
            clf = SVC()
            clf.fit(X, Y, kernel, C, kernel_kwargs)
            self.clfs.append([clf, pos, neg])

    def make_data(self, Xtr, Ytr, pos, neg):
        Y = -np.ones(Ytr.shape)
        Y[np.where(Ytr == pos)] = 1
        idx = np.where((Ytr == pos) | (Ytr == neg))
        return Xtr[idx], Y[idx]

    def predict(self, Xte):
        N = Xte.shape[0]
        scores = np.zeros((N,self.n_classes))
        for j in range(len(self.clfs)):
            clf, pos, neg = self.clfs[j]
            pred = clf.predict(Xte)
            scores[np.where(pred == 1), pos] += 1
            scores[np.where(pred == -1), neg] += 1
        Yte = np.argmax(scores, axis=1)
        return Yte


class OvAKSVC():
    def __init__(self):
        self.clfs = []

    def fit(self, Xtr, Ytr, kernel, C, kernel_kwargs):
        self.n_classes = len(np.unique(Ytr))
        for c in tqdm(range(self.n_classes)):
            #print('Training:', c)
            X, Y = self.make_data(Xtr, Ytr, c)
            clf = SVC()
            clf.fit(X, Y, kernel, C, kernel_kwargs)
            self.clfs.append(clf)

    def make_data(self, Xtr, Ytr, c):
        Y = np.zeros(Ytr.shape)
        c_idx = np.where(Ytr == c)
        Y[c_idx] = 1
        sample_idx = tuple([random.sample(list(np.where(Y != 1)[0]), k=len(c_idx[0]))])
        Y[sample_idx] = -1
        idx = np.where(Y != 0)
        return Xtr[idx], Y[idx]

    def predict(self, Xte):
        N = Xte.shape[0]
        scores = np.zeros((N,self.n_classes))
        for c in range(len(self.clfs)):
            clf = self.clfs[c]
            pred = clf.separating_function(Xte) + clf.b
            scores[:,c] = pred
        Yte = np.argmax(scores, axis=1)
        return Yte

class KSVC():
    def __init__(self):
        pass

    def fit(self, Xtr, Ytr, kernel, C, decision_function, kernel_kwargs):
        if decision_function == 'ovo':
            self.SVC = OvOKSVC()
        elif decision_function == 'ova':
            self.SVC = OvAKSVC()
        self.SVC.fit(Xtr, Ytr, kernel, C, kernel_kwargs)

    def predict(self, Xte):
        return self.SVC.predict(Xte)
