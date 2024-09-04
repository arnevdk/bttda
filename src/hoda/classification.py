import math
import pdb

import numpy as np
import tensorly as tl
from kneed import KneeLocator
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

from hoda.backend import std
from hoda.tensorize import vec
from hoda.util import f_oneway

try:
    from toeplitzlda.classification import ToeplitzLDA
except ImportError:
    pass


class ZScore(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_ = tl.mean(X, axis=0)
        self.std_ = std(X, axis=0)
        return self

    def transform(self, X, y=None):
        return (X - self.mean_) / self.std_


class SelectF(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=None, verbose=False):
        self.alpha = alpha
        self.verbose = verbose

    def fit(self, X, y=None):
        n_samples, *_ = X.shape
        self.F_, self.p_ = f_oneway(X, y)
        if self.alpha is None:
            F = tl.to_numpy(self.F_)
            if len(F) > 2:
                kneedle = KneeLocator(
                    np.arange(len(F)), sorted(F), curve="convex", direction="increasing"
                )
                self.mask_ = self.F_ >= kneedle.knee_y
            else:
                self.mask_ = tl.ones(len(F), dtype=bool)
        else:
            self.mask_ = self.p_ < self.alpha
        if not np.any(self.mask_):
            self.mask_[np.argmax(self.F_)] = True
        if self.verbose:
            print(f"Selected {np.count_nonzero(self.mask_)}/{len(self.mask_)} features")
        return self

    def transform(self, X, y=None):
        return X[:, self.mask_]


class ToeplitzLDAWrapper(BaseEstimator, ClassifierMixin):
    def fit(self, X, y=None):
        self.classes_ = np.unique(y)
        n_epochs, n_channels, n_samples = X.shape
        self.tlda_ = ToeplitzLDA(n_channels=n_channels, data_is_channel_prime=False)
        return self.tlda_.fit(vec(X), y)

    def decision_function(self, X):
        n_epochs, n_channels, n_samples = X.shape
        return self.tlda_.decision_function(vec(X))

    def predict(self, X):
        return self.tlda_.predict(vec(X))

    def predict_proba(self, X):
        return self.tlda_.predict_proba(vec(X))
