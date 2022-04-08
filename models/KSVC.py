#!/usr/bin/env python

# file management libraries
import importlib

# numerical libraries
import random
import numpy as np
from scipy import optimize

# dependencies
from kernels import *

# Kernel Support Vector Classifier
class SVC():
    '''
    Support Vector Classifier.

    Arguments:
        - alpha: np.array
            separating function coefficients
        - epsilon: float
            tolerance term for optimization
    '''

    def __init__(self, epsilon = 1e-3):

        # set SVC parameters
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
                regularization parameter
            - kernel_kwargs: dict
                arguments to pass to kernel class
        '''

        # set data
        self.Xtr = Xtr

        # set parameters
        self.C = C
        self.kernel_class = getattr(importlib.import_module("kernels"), kernel)
        self.kernel = self.kernel_class(**kernel_kwargs)

        # compute Gram matrix
        K = self.kernel.compute(self.Xtr, self.Xtr)

        # define useful variables
        N = len(Ytr)
        diag_y = np.diag(Ytr)

        # define lagrange dual problem
        def loss(alpha):

            # set dual Loss
            return 1/2*np.linalg.multi_dot([alpha.T, diag_y, K, diag_y, alpha])\
                    - alpha.T.dot(np.ones(N))

        # define partial derivate of loss on alpha
        def grad_loss(alpha):

            # partial derivative of the dual loss w.r.t. alpha
            return np.linalg.multi_dot([diag_y, K, diag_y, alpha]) - np.ones(N)


        # set equality constraints
        fun_eq = lambda alpha: -alpha.T.dot(Ytr)
        jac_eq = lambda alpha: -Ytr

        # set inequality constraints
        fun_ineq = lambda alpha: self.C*np.ones(N) - alpha
        jac_ineq = lambda alpha: - np.eye(N)
        fun_ineq_pos = lambda alpha: alpha
        jac_ineq_pos = lambda alpha: np.eye(N)

        # set contraints in dictionary
        constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq', 'fun': fun_ineq , 'jac': jac_ineq},
                       {'type': 'ineq', 'fun': fun_ineq_pos, 'jac': jac_ineq_pos})

        # optimize
        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N),
                                   method='SLSQP',
                                   jac=lambda alpha: grad_loss(alpha),
                                   constraints=constraints)

        # save optimized value for alpha
        self.alpha = optRes.x

        # select indices of support vectors
        supportIndices = np.where((self.alpha > self.epsilon)\
                                  & (C*np.ones(N) - self.alpha > self.epsilon))

        # compute diag(y).dot(alpha) to avoid computing it at inference
        self.alpha = diag_y.dot(self.alpha)

        # compute offset of the classifier
        self.b = np.mean(Ytr[supportIndices] - self.alpha.T.dot(K)[supportIndices])

    # separating function
    def separating_function(self,x):

        # return separating criterion
        return self.alpha.T.dot(self.kernel.compute(self.Xtr,x))

    # prediction function
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

        # compute separating function
        d = self.separating_function(Xte)

        # compute predictions
        Yte = 2 * (d+self.b> 0) - 1

        # return predictions
        return Yte

class OvOKSVC():
    '''
    One vs. One Kernel Support Vector Classifier.
    '''

    def __init__(self):

        # set list of classifiers
        self.clfs = []

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
            - kernel_kwargs: dict
                arguments to pass to kernel class
        '''

        # set number of classes
        self.n_classes = len(np.unique(Ytr))

        # initialise all combinations
        comb = [(pos,neg) for pos in range(self.n_classes) for neg in range(pos+1, self.n_classes)]

        # compute SVC over all combinations
        for i, (pos, neg) in enumerate(comb):

            print(f'Training {pos} vs {neg} ({i+1}/{len(comb)})...')

            # create subsampled datasets
            X, Y = self.make_data(Xtr, Ytr, pos, neg)

            # initialise SVC
            clf = SVC()

            # fit SVC
            clf.fit(X, Y, kernel, C, kernel_kwargs)

            # append SVC for combination to list of combination-wise SVCs
            self.clfs.append([clf, pos, neg])

    def make_data(self, Xtr, Ytr, pos, neg):

        # get indices of considered classes
        Y = -np.ones(Ytr.shape)
        Y[np.where(Ytr == pos)] = 1
        idx = np.where((Ytr == pos) | (Ytr == neg))

        # return subsampled datasets
        return Xtr[idx], Y[idx]

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

        # set number of samples
        N = Xte.shape[0]

        # initialise scores
        scores = np.zeros((N,self.n_classes))

        # compute vector for decision function (voting classifier)
        for j in range(len(self.clfs)):
            clf, pos, neg = self.clfs[j]
            pred = clf.predict(Xte)
            scores[np.where(pred == 1), pos] += 1
            scores[np.where(pred == -1), neg] += 1

        # obtain class with highest voting score
        Yte = np.argmax(scores, axis=1)

        # return predicted classes
        return Yte

class OvRKSVC():
    '''
    One vs. Rest Kernel Support Vector Classifier.
    '''

    def __init__(self):

        # set list of classifiers
        self.clfs = []

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
            - kernel_kwargs: dict
                arguments to pass to kernel class
        '''

        # set number of classes
        self.n_classes = len(np.unique(Ytr))

        # compute SVC over all classes
        for c in range(self.n_classes):

            print(f'Training {c} vs rest ({c+1}/{self.n_classes})...')

            # create subsampled datasets
            X, Y = self.make_data(Xtr, Ytr, c)

            # initialise SVC
            clf = SVC()

            # fit SVC
            clf.fit(X, Y, kernel, C, kernel_kwargs)

            # append SVC for combination to list of classes-wise SVCs
            self.clfs.append(clf)

    def make_data(self, Xtr, Ytr, c):

        # get indices of considered classes
        Y = np.zeros(Ytr.shape)
        c_idx = np.where(Ytr == c)
        Y[c_idx] = 1

        # sample from the rest of data to balance classes
        sample_idx = tuple([random.sample(list(np.where(Y != 1)[0]), k=len(c_idx[0]))])
        Y[sample_idx] = -1
        idx = np.where(Y != 0)

        # return subsampled datasets
        return Xtr[idx], Y[idx]

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

        # set number of samples
        N = Xte.shape[0]

        # initialise scores
        scores = np.zeros((N,self.n_classes))

        # compute vector for decision function (voting classifier)
        for c in range(len(self.clfs)):
            clf = self.clfs[c]
            pred = clf.separating_function(Xte) + clf.b
            scores[:,c] = pred

        # obtain class with highest voting score
        Yte = np.argmax(scores, axis=1)

        # return predicted classes
        return Yte

class KSVC():
    '''
    Kernel Support Vector Classifier.
    '''

    def __init__(self):
        pass

    def fit(self, Xtr, Ytr, kernel, C, decision_function, kernel_kwargs):
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
            - decision_function: str
                OvO ("ovo) or OvA ("ova") decision function
            - kernel_kwargs: dict
                arguments to pass to kernel class
        '''

        # set OvO KSVC
        if decision_function == 'ovo':
            self.SVC = OvOKSVC()

        # set OvR KSVC
        elif decision_function == 'ovr':
            self.SVC = OvRKSVC()

        else:
            raise NotImplementedError("Please select a valid decision function i.e. 'ovr' or 'ova'")

        # fit KSVC
        self.SVC.fit(Xtr, Ytr, kernel, C, kernel_kwargs)

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

        # compute prediction
        Yte = self.SVC.predict(Xte)

        # return prediction
        return Yte
