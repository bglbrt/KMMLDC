#!/usr/bin/env python



# file management libraries
import importlib
from tqdm import tqdm
import random

# numerical libraries
import numpy as np
from scipy import optimize
from sklearn.svm import SVC

# dependencies
from kernels import *

# Support Vector Classifier
class skSVC():
    '''
    Support Vector Classifier.
    '''

    def __init__(self, epsilon = 1e-3):      
        self.alpha = None
        self.epsilon = epsilon

    def fit(self, Xtr, Ytr, kernel, C, decision_function, kernel_kwargs):
        self.SVC = SVC()
        self.SVC.fit(Xtr,Ytr)

    def predict(self, Xte):
        return self.SVC.predict(Xte)