#!/usr/bin/env python

# numerical libraries
import numpy as np
import pandas as pd

from scipy.ndimage import uniform_filter

# image to HOG transformation
def image_to_hog(X):
    '''
    Computes the hog of an image inputted as vector.

    Arguments:
        - X: np.array
            RGB image as one-dimensional vector

    Returns:
        - HOG: np.array
            HOG of input image
    '''

    # split original array into RGB channels and into 32x32 arrays
    X_R = X[:1024].reshape((32, 32))
    X_G = X[1024:2048].reshape((32, 32))
    X_B = X[2048:3072].reshape((32, 32))

    # initialise final HOG vector
    HOG = np.zeros(1536)

    # populate dictionary for each channel
    for C, X in enumerate([X_R, X_G, X_B]):

        # compute gradient in both axes
        G_X = np.zeros((32, 32))
        G_Y = np.zeros((32, 32))
        G_X[:, :-1] = np.diff(X, n=1, axis=1)
        G_Y[:-1, :] = np.diff(X, n=1, axis=0)

        # compute gradient magnitude
        G_M = np.sqrt(np.square(G_X) + np.square(G_Y))

        # compute gradient orientation
        G_O = np.arctan2((G_Y + 1e-15), (G_X + 1e-15)) * (180 / np.pi) + 90

        # initialise HOG matrix
        HOG_MAT = np.zeros((8, 8, 8))

        # iterate over orientations
        for i in range(8):

            # select orientations in given range
            Oi = np.where(G_O < (180 / 8) * (i + 1), G_O, 0)
            Oi = np.where(G_O >= (180 / 8) * i, Oi, 0)

            # select magnitudes for those orientations
            Mi = np.where(Oi > 0, G_M, 0)

            # fill matrix with magnitudes
            HOG_MAT[:,:,i] = uniform_filter(Mi, size=(4, 4))[2::4, 2::4].T

        # fill HOG vector
        HOG[512*C:512*(C+1)] = HOG_MAT.flatten()

    # return hog
    return HOG

def array_to_hog(X):
    '''
    Computes the HOG of multiple images in an array.

    Arguments:
        - X: np.array
            input feature matrix

    Returns:
        - X_HOG: np.array
            HOG feature matrix
    '''

    # compute HOG along axis
    X_HOG = np.apply_along_axis(lambda z: image_to_hog(z), 1, X)

    # return HOG
    return X_HOG

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

    def __init__(self, x_train_path, x_test_path, y_train_path, transform=None):

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
        if self.transform is None:
            pass

        elif self.transform == 'histogram':
            self.Xtr = array_to_hist(self.Xtr)
            self.Xte = array_to_hist(self.Xte)

        elif self.transform == 'hog':
            self.Xtr = array_to_hog(self.Xtr)
            self.Xte = array_to_hog(self.Xte)

        else:
            raise NotImplementedError("Transform method not implemented!")

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
