import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pybaselines.misc import beads
from pybaselines.morphological import mormol, rolling_ball
from pybaselines.whittaker import arpls, asls
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
from scipy.signal import savgol_filter


class BaselineCorrector(BaseEstimator, TransformerMixin):
    def __init__(self, method="asls"):
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if type(X) != np.ndarray:
            X = np.asarray(X)

        bl = np.zeros_like(X)

        if self.method == "asls":
            for i, row in enumerate(X):
                bl[i] = asls(row)[0]

        elif self.method == "arpls":
            for i, row in enumerate(X):
                bl[i] = arpls(row)[0]

        elif self.method == "mormol":
            for i, row in enumerate(X):
                bl[i] = mormol(row)[0]

        elif self.method == "rolling ball":
            for i, row in enumerate(X):
                bl[i] = rolling_ball(row)[0]

        elif self.method == "beads":
            for i, row in enumerate(X):
                bl[i] = beads(row)[0]

        else:
            raise ValueError(f"Method {self.method} does not exist.")


        return X - bl


class ColumnSelectorPCA(BaseEstimator, TransformerMixin):
    """Class to select a range of components from PCA, so that components do not 
    have to be calculated over and over."""

    def __init__(self, n_components=None):
        """Initialize range of components."""
        if n_components == None:
            n_components = -1
        self.n_components = n_components
    
    def fit(self, X, y=None):
        """Does not do anything, included for compatibility with pipelines"""
        return self

    def transform(self, X, y=None):
        """Return the previously selected range of components."""
        return X[:, 0:self.n_components]


class OrderedLabelEncoder(LabelEncoder):
    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self


class RangeLimiter(BaseEstimator, TransformerMixin):
    def __init__(self, min_index=0, max_index=None):
        self.min_index = min_index
        self.max_index = max_index
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.max_index == None:
            self.max_index = X.shape[1]

        X_red  = X[:, self.min_index:self.max_index]
        X_red = (X_red.T - X_red.min(axis=1)).T

        return X_red

class SavGolFilter(BaseEstimator, TransformerMixin):
    """Class to smooth spectral data using a Savitzky-Golay Filter."""

    def __init__(self, window=15, poly=3):
        """Initialize window size and polynomial order of the Savitzky-Golay Filter"""
        self.window = window
        self.poly = poly

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_smooth = savgol_filter(X, window_length=self.window, polyorder=self.poly)
        return X_smooth


def wn_range(wns, min_wn=None, max_wn=None):
    if min_wn:
        min_i = np.where(wns >= min_wn)[0][0]
    else:
        min_i = 0

    if max_wn:
        max_i = np.where(wns <= max_wn)[0][-1]
    else:
        max_i = len(wns)
    
    return min_i, max_i