import numpy as np
import tensorly as tl
import tensorly.decomposition
import tensorly.tenalg
from sklearn.base import BaseEstimator, TransformerMixin


class MLSVD(BaseEstimator, TransformerMixin):
    def __init__(self, modes=None, rank=None):
        self.modes = modes
        self.rank = rank

    def fit(self, X, y=None):
        shape = X.shape[1:]
        order = len(shape)
        modes = self.modes
        if modes is None:
            modes = np.arange(order)
        else:
            modes = np.asarray(modes)

        self.factors_ = [tl.eye(shape[k]) for k in range(order)]
        _, factors = tensorly.decomposition.partial_tucker(
            X, modes=modes + 1, rank=self.rank
        )
        for mi, mode in enumerate(modes):
            self.factors_[mode] = factors[mi]
        return self

    def transform(self, X, y=None):
        modes = tuple(range(1, len(X.shape)))
        Xt = tensorly.tenalg.multi_mode_dot(
            X, self.factors_, modes=modes, transpose=True
        )
        return Xt
