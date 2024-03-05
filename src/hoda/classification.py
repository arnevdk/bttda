import numpy as np
import tensorly as tl
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import FunctionTransformer

try:
    from toeplitzlda.classification import ToeplitzLDA
except ImportError:
    pass


def vec(X, y=None, extra=None):
    x = X.reshape((X.shape[0], -1))
    return tl.to_numpy(x)


class Vectorize(FunctionTransformer):
    def __init__(self, **params):
        super().__init__(func=vec, **params)


class SelectF(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def fit(self, X, y=None):
        self.F_, self.p_ = f_classif(X, y)
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
