import ipdb
import numpy as np
import scipy.linalg
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer


def hankel_tensor(X, y=None):
    n_times = X.shape[-1]
    half = n_times // 2 + 1

    def _hankel_nopad(x):
        return scipy.linalg.hankel(x[:half], x[half - 1 :])

    return np.apply_along_axis(_hankel_nopad, -1, X)


def hankel_tensor_inv(Xh, y=None):
    X1 = Xh[:, :, 0, :]
    X2 = Xh[:, :, -1, :]
    if X2.shape[-1] % 2:
        X2 = X2[:, :, 1:]
    return np.concatenate((X1, X2), axis=-1)


class HankelTensor(FunctionTransformer):
    def __init__(**params):
        params["func"] = hankel_tensor
        super().__init__(**params)


class STFTensor(TransformerMixin):
    def __init__(**params):
        raise NotImplementedError
