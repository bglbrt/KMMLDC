#!/usr/bin/env python

# numerical libraries
import numpy as np
import pandas as pd

# computer vision libraries
from scipy.ndimage import uniform_filter
from scipy.ndimage import rotate

# image to HOG transformation
def image_to_hog(X, cells_size, n_orientations):
    '''
    Computes the hog of an image inputted as vector.

    Arguments:
        - X: np.array
            RGB image as one-dimensional vector
        - cells_size: int
            size of cells for HOG transform
        - n_orientations: int
            number of histogram orientations for HOG transform

    Returns:
        - HOG: np.array
            HOG of input image
    '''

    # split original array into RGB channels and into 32x32 arrays
    X_R = X[:1024].reshape((32, 32))
    X_G = X[1024:2048].reshape((32, 32))
    X_B = X[2048:3072].reshape((32, 32))

    # initialise final HOG vector
    HOG = np.zeros((32//cells_size)**2 * n_orientations * 3)

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
        HOG_MAT = np.zeros((32//cells_size, 32//cells_size, n_orientations))

        # iterate over orientations
        for i in range(n_orientations):

            # select orientations in given range
            Oi = np.where(G_O < (180 / n_orientations) * (i + 1), G_O, 0)
            Oi = np.where(G_O >= (180 / n_orientations) * i, Oi, 0)

            # select magnitudes for those orientations
            Mi = np.where(Oi > 0, G_M, 0)

            # fill matrix with magnitudes
            HOG_MAT[:,:,i] = uniform_filter(Mi, size=(cells_size, cells_size))[int(cells_size/2)::cells_size, int(cells_size/2)::cells_size].T

        # fill HOG vector
        HOG[((32//cells_size)**2 * n_orientations)*C:((32//cells_size)**2 * n_orientations)*(C+1)] = HOG_MAT.flatten()

    # return hog
    return HOG

def array_to_hog(X, cells_size, n_orientations):
    '''
    Computes the HOG of multiple images in an array.

    Arguments:
        - X: np.array
            input feature matrix
        - cells_size: int
            size of cells for HOG transform
        - n_orientations: int
            number of histogram orientations for HOG transform

    Returns:
        - X_HOG: np.array
            HOG feature matrix
    '''

    # compute HOG along axis
    X_HOG = np.apply_along_axis(lambda z: image_to_hog(z, cells_size, n_orientations), 1, X)

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

def flip_image(X):
    '''
    Flips an image inputted as vector.

    Arguments:
        - X: np.array
            RGB image as one-dimensional vector

    Returns:
        - X_FLIPPED: np.array
            flipped RGB image as one-dimensional vector
    '''

    # split original array into RGB channels and into 32x32 arrays
    X_R = X[:1024].reshape((32, 32))
    X_G = X[1024:2048].reshape((32, 32))
    X_B = X[2048:3072].reshape((32, 32))

    # initialise flipped image
    X_FLIPPED = np.zeros(3072)

    # populate flipped image
    X_FLIPPED[:1024] = np.flip(X_R, axis=1).reshape(1024)
    X_FLIPPED[1024:2048] = np.flip(X_G, axis=1).reshape(1024)
    X_FLIPPED[2048:3072] = np.flip(X_B, axis=1).reshape(1024)

    # return flipped image
    return X_FLIPPED

def rotate_image(X, angle):
    '''
    Rotates an image inputted as vector.

    Arguments:
        - X: np.array
            RGB image as one-dimensional vector
        - angle: float
            angle for rotation (in degrees)

    Returns:
        - X_ROTATED: np.array
            flipped RGB image as one-dimensional vector
    '''

    # split original array into RGB channels and into 32x32 arrays
    X_R = X[:1024].reshape((32, 32))
    X_G = X[1024:2048].reshape((32, 32))
    X_B = X[2048:3072].reshape((32, 32))

    # initialise flipped image
    X_ROTATED = np.zeros(3072)

    # populate flipped image
    X_ROTATED[:1024] = rotate(X_R, angle, mode='reflect', reshape=False).reshape(1024)
    X_ROTATED[1024:2048] = rotate(X_G, angle, mode='reflect', reshape=False).reshape(1024)
    X_ROTATED[2048:3072] = rotate(X_B, angle, mode='reflect', reshape=False).reshape(1024)

    # return flipped image
    return X_ROTATED

def augment_horizontal(X):
    '''
    Adds all horizontally-flipped versions of images to data.

    Arguments:
        - X: np.array
            input feature matrix

    Returns:
        - X_AUGMENTED: np.array
            augmented feature matrix
    '''

    # get dataset of flipped images
    X_FLIPPED = np.apply_along_axis(lambda z: flip_image(z), 1, X)

    # concatenate unflipped and flipped images
    X_AUGMENTED = np.concatenate([X, X_FLIPPED])

    # return augmented feature matrix
    return X_AUGMENTED

def augment_rotate(X, angle):
    '''
    Adds all rotated versions of images to data.

    Arguments:
        - X: np.array
            input feature matrix

    Returns:
        - X_ROTATED: np.array
            augmented feature matrix
    '''

    # get dataset of flipped images
    X_ROTATED_RIGHT = np.apply_along_axis(lambda z: rotate_image(z, angle), 1, X)
    X_ROTATED_LEFT = np.apply_along_axis(lambda z: rotate_image(z, -angle), 1, X)

    # concatenate unflipped and flipped images
    X_AUGMENTED = np.concatenate([X, X_ROTATED_RIGHT, X_ROTATED_LEFT])

    # return augmented feature matrix
    return X_AUGMENTED

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

    def __init__(self, x_train_path, x_test_path, y_train_path, augment, angle, transform, cells_size, n_orientations):

        if cells_size % 2 != 0:
            raise Error('Error! Cells size for HOG transform must be even.')

        # set transform, cell_size and augment
        self.augment = augment
        self.angle = angle
        self.transform = transform
        self.cells_size = cells_size
        self.n_orientations = n_orientations

        # set paths
        self.x_train_path = x_train_path
        self.X_test_path = x_test_path
        self.y_train_path = y_train_path

        # load data
        self.Xtr = np.array(pd.read_csv(x_train_path, header=None, sep=',', usecols=range(3072)))
        self.Xte = np.array(pd.read_csv(x_test_path, header=None, sep=',', usecols=range(3072)))
        self.Ytr = np.array(pd.read_csv(y_train_path, sep=',', usecols=[1])).squeeze()

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

        # compute data augmentation
        if self.augment is None:
            pass

        elif self.augment == 'horizontal':
            self.Xtr = augment_horizontal(self.Xtr)
            self.Ytr = np.concatenate([self.Ytr, self.Ytr])

        elif self.augment == 'rotate':
            self.Xtr = augment_rotate(self.Xtr, self.angle)
            self.Ytr = np.concatenate([self.Ytr, self.Ytr, self.Ytr])

        elif self.augment == 'all':
            self.Xtr = augment_rotate(self.Xtr, self.angle)
            self.Xtr = augment_horizontal(self.Xtr)
            self.Ytr = np.concatenate([self.Ytr, self.Ytr, self.Ytr, self.Ytr, self.Ytr, self.Ytr])

        # compute transform
        if self.transform is None:
            pass

        elif self.transform == 'histogram':
            self.Xtr = array_to_hist(self.Xtr)
            self.Xte = array_to_hist(self.Xte)

        elif self.transform == 'hog':
            self.Xtr = array_to_hog(self.Xtr, self.cells_size, self.n_orientations)
            self.Xte = array_to_hog(self.Xte, self.cells_size, self.n_orientations)

        else:
            raise NotImplementedError("Transform method not implemented!")

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
        o = np.ones(int(split_size*self.Xtr.shape[0]))
        z = np.zeros(self.Xtr.shape[0] - int(split_size*self.Xtr.shape[0]))
        val_split = np.concatenate([o, z]).astype(bool)
        np.random.shuffle(val_split)

        # split data in train and validation data
        self.Xval = self.Xtr[val_split]
        self.Xtra = self.Xtr[~val_split]
        self.Yval = self.Ytr[val_split]
        self.Ytra = self.Ytr[~val_split]

        # compute data augmentation
        if self.augment is None:
            pass

        elif self.augment == 'horizontal':
            self.Xtra = augment_horizontal(self.Xtra)
            self.Ytra = np.concatenate([self.Ytra, self.Ytra])

        elif self.augment == 'rotate':
            self.Xtra = augment_rotate(self.Xtra, self.angle)
            self.Ytra = np.concatenate([self.Ytra, self.Ytra, self.Ytra])

        elif self.augment == 'all':
            self.Xtra = augment_rotate(self.Xtra, self.angle)
            self.Xtra = augment_horizontal(self.Xtra)
            self.Ytra = np.concatenate([self.Ytra, self.Ytra, self.Ytra, self.Ytra, self.Ytra, self.Ytra])

        # compute transform
        if self.transform is None:
            pass

        elif self.transform == 'histogram':
            self.Xtra = array_to_hist(self.Xtra)
            if split_size > 0:
                self.Xval = array_to_hist(self.Xval)

        elif self.transform == 'hog':
            self.Xtra = array_to_hog(self.Xtra, self.cells_size, self.n_orientations)
            if split_size > 0:
                self.Xval = array_to_hog(self.Xval, self.cells_size, self.n_orientations)

        else:
            raise NotImplementedError("Transform method not implemented!")

        # return data
        return self.Xtra, self.Xval, self.Ytra, self.Yval
