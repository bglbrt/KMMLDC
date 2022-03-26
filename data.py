#!/usr/bin/env python

# numerical libraries
import numpy as np
import pandas as pd

# data loader
class LOADER():
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

    def __init__(self):

        # set paths
        self.x_train_path = x_train_path
        self.X_test_path = x_test_path
        self.y_train_path = y_train_path

        # load data
        self.Xtr = np.array(pd.read_csv(x_train_path, header=None, sep=',', usecols=range(3072)))
        self.Xte = np.array(pd.read_csv(x_test_path, header=None, sep=',', usecols=range(3072)))
        self.Ytr = np.array(pd.read_csv(y_train_path, sep=',', usecols=[1])).squeeze()

    def load_train_test():
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

    def load_train_val(split_size=.2):
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
        Xval = self.Xtr[val_split].shape[0]
        Xtra = self.Xtr[~val_split].shape[0]
        Yval = self.Ytr[val_split].shape[0]
        Ytra = self.Ytr[~val_split].shape[0]

        # return data
        return Xtra, Xval, Ytra, Yval
