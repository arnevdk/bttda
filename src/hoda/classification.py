import numpy as np
import tensorly as tl
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectFwe
from sklearn.preprocessing import FunctionTransformer

# from toeplitzlda.classification import ToeplitzLDA


def reshape(X, y=None):
    return tl.to_numpy(X.reshape((X.shape[0], -1)))


class Vectorize(FunctionTransformer):
    def __init__(self, **params):
        super().__init__(func=reshape, **params)


class SelectFweAtLeastOne(SelectFwe):
    def _get_support_mask(self):
        mask = super()._get_support_mask()
        if not np.any(mask):
            mask = self.pvalues_ == np.min(self.pvalues_)
        return mask


# class ToeplitzLDAWrapper(BaseEstimator, ClassifierMixin):
#    def fit(self, X, y=None):
#        self.classes_ = np.unique(y)
#        n_epochs, n_channels, n_samples = X.shape
#        self.tlda_ = ToeplitzLDA(n_channels=n_channels, data_is_channel_prime=False)
#        return self.tlda_.fit(reshape(X), y)
#
#    def decision_function(self, X):
#        n_epochs, n_channels, n_samples = X.shape
#        return self.tlda_.decision_function(reshape(X))
#
#    def predict(self, X):
#        return self.tlda_.predict(reshape(X))
#
#    def predict_proba(self, X):
#        return self.tlda_.predict_proba(reshape(X))
