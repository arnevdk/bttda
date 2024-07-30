import numpy as np
import tensorly as tl
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
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def fit(self, X, y=None):
        self.F_, self.p_ = f_oneway(X, y)
        self.mask_ = self.p_ < self.alpha
        if not np.any(self.mask_):
            self.mask_[np.argmax(self.F_)] = True
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
