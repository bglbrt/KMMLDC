#!/usr/bin/env python

# numerical libraries
import numpy as np
import pandas as pd

# array to histogram transformation
def array_to_hist(X):
    '''
    Computes the histograms of a features matrix.

    Arguments:
        - X: np.array
            input feature matrix

    Returns:
        - X_HIST: np.array
            histogram feature matrix
    '''

    # split original array into RGB channels
    X_R = X[:, :1024]
    X_G = X[:, 1024:2048]
    X_B = X[:, 2048:3072]

    # compute histogram over each channel
    X_R_H = (1/1024) * np.apply_along_axis(lambda z: np.histogram(z, range=(-.5, .5), bins=256)[0], 1, X_R)
    X_G_H = (1/1024) * np.apply_along_axis(lambda z: np.histogram(z, range=(-.5, .5), bins=256)[0], 1, X_G)
    X_B_H = (1/1024) * np.apply_along_axis(lambda z: np.histogram(z, range=(-.5, .5), bins=256)[0], 1, X_B)

    # concatenate matrix
    X_HIST = np.concatenate([X_R_H, X_G_H, X_B_H], axis=1)

    # return histogram matrix
    return X_HIST

# data loader
class Loader():
    '''
    Data loading class.

    Arguments:
        - x_train_path: str
            path to X train file
        - x_test_path: str
            path to X test file
        - y_train_path: str
            path to y train file
    '''

    def __init__(self, x_train_path, x_test_path, y_train_path, transform):

        # set transform
        self.transform = transform

        # set paths
        self.x_train_path = x_train_path
        self.X_test_path = x_test_path
        self.y_train_path = y_train_path

        # load data
        self.Xtr = np.array(pd.read_csv(x_train_path, header=None, sep=',', usecols=range(3072)))
        self.Xte = np.array(pd.read_csv(x_test_path, header=None, sep=',', usecols=range(3072)))
        self.Ytr = np.array(pd.read_csv(y_train_path, sep=',', usecols=[1])).squeeze()

        # compute transform
        if self.transform == 'histogram':
            self.Xtr = array_to_hist(self.Xtr)
            self.Xte = array_to_hist(self.Xte)

    def load_train_test(self):
        '''
        Load train and test data as such.

        Returns:
            - Xtr: np.array
                X train data
            - Xte: np.array
                X test data
            - Ytr: np.array
                y train data
        '''

        # return data
        return self.Xtr, self.Xte, self.Ytr

    def load_train_val(self, split_size=.2):
        '''
        Load train and test data as such.

        Arguments:
            - split_size: float
                share of data in validation data

        Returns:
            - Xtr: np.array
                X train data
            - Xval: np.array
                X validation data
            - Ytr: np.array
                y train data
            - Yval: np.array
                y validation data
        '''

        # create splitting mask
        o = np.ones(int(0.2*self.Xtr.shape[0]))
        z = np.zeros(self.Xtr.shape[0] - int(0.2*self.Xtr.shape[0]))
        val_split = np.concatenate([o, z]).astype(bool)
        np.random.shuffle(val_split)

        # split data in train and validation data
        Xval = self.Xtr[val_split]
        Xtra = self.Xtr[~val_split]
        Yval = self.Ytr[val_split]
        Ytra = self.Ytr[~val_split]

        # return data
        return Xtra, Xval, Ytra, Yval
