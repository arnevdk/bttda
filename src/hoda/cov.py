# import numpy as np
import math
import pdb
import warnings

import numpy as np
import scipy.linalg
import tensorly as tl
from sklearn.base import BaseEstimator

from hoda.util import get_eye

try:
    import cupyx.scipy.linalg
except ImportError:
    pass


def mode_scatter(
    X, k, weights=None, shrinkage=0, toeplitz=None, taper=False, assume_centered=False
):
    """Calculate the scatter matrix along a given tensor mode"""

    n_samples, *shape = X.shape
    order = len(shape)
    n_features = shape[k]
    # Determine mode scatter
    modes = [0] + [kk + 1 for kk in range(order) if kk != k]
    if not assume_centered:
        X = X - tl.mean(X, axis=0)

    if weights is None:
        weights = tl.ones(n_samples)
    weights = tl.reshape(
        weights, (n_samples,) + (1,) * (X.ndim - 1)
    )  # Expands to match X

    # if toeplitz is not None and k in toeplitz:
    #    n_lags = shape[k]
    #    scatter_toep = tl.zeros(n_lags)
    #    Xt = tl.moveaxis(X, k+1, 1)
    #    for lag in range(n_lags):
    #        valid = Xt[:, :n_lags-lag]* Xt[:, lag:]*weights
    #        scatter_toep[lag] = tl.sum(valid)
    #    scatter_toep/=n_lags
    #    if tl.get_backend()=='numpy':
    #        scatter =  scipy.linalg.toeplitz(scatter_toep)
    #    elif tl.get_backend()=='cupy':
    #        scatter =  cupyx.scipy.linalg.toeplitz(scatter_toep)

    # else:
    #    scatter = tl.tenalg.tensordot(X*weights, X,  (modes,modes))
    scatter = tl.tenalg.tensordot(X * weights, X, (modes, modes))
    if toeplitz is not None and k in toeplitz:
        scatter = force_toeplitz(scatter, taper=taper)

    # Determine shrinkage
    if shrinkage == "lw":
        if scatter.shape[0] == 1:
            shrinkage = 0
        else:
            Xf = tl.unfold(X, k + 1)
            shrinkage = ledoit_wolf_shrinkage(
                Xf.T,
                assume_centered=assume_centered,
            )
    elif shrinkage == "oas":
        n = n_samples * math.prod(shape) / shape[k]
        cov = scatter / (n - 1)
        shrinkage = oas(cov, n)
    elif shrinkage == "ss":
        shrinkage = schaefer_strimmer_shrinkage(X, k)
    elif shrinkage == "ell1":
        raise NotImplementedError
    elif shrinkage == "ell2":
        raise NotImplementedError
    elif shrinkage == "ell3":
        raise NotImplementedError
    elif shrinkage == "loocv":
        raise NotImplemented

    trace = tl.trace(scatter)
    scale = trace / n_features
    target = get_eye(n_features) * scale
    scatter = (1 - shrinkage) * scatter + shrinkage * target
    return scatter, shrinkage


def force_toeplitz(A, taper=False):
    n, _ = A.shape
    toep = tl.zeros(n)
    for i in range(n):
        diag = tl.diag(A, k=i)
        toep[i] = tl.mean(diag)

    if taper:
        taper = tl.arange(len(toep), 0, -1) - 1
        print(taper)
        toep = toep * taper
    if tl.get_backend() == "numpy":
        return scipy.linalg.toeplitz(toep)
    elif tl.get_backend() == "cupy":
        return cupyx.scipy.linalg.toeplitz(toep)


def ledoit_wolf_shrinkage(
    X,
    assume_centered=False,
    block_size=1000,
):
    """Estimate the shrunk Ledoit-Wolf covariance matrix.
    Read more in the :ref:`User Guide <shrunk_covariance>`.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data from which to compute the Ledoit-Wolf shrunk covariance shrinkage.
    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, data will be centered before computation.
    block_size : int, default=1000
        Size of blocks into which the covariance matrix will be split.
    Returns
    -------
    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.
    Notes
    -----
    The regularized (shrunk) covariance is:
    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)
    where mu = trace(cov) / n_features
    """
    # for only one feature, the result is the same whatever the shrinkage
    if len(X.shape) == 2 and X.shape[1] == 1:
        return 0.0
    if X.ndim == 1:
        X = tl.reshape(X, (1, -1))

    if X.shape[0] == 1:
        warnings.warn(
            "Only one sample available. You may want to reshape your data array"
        )
    n_samples, n_features = X.shape

    # optionally center data
    if not assume_centered:
        X = X - X.mean(0)

    # A non-blocked version of the computation is present in the tests
    # in tests/test_covariance.py

    # number of blocks to split the covariance matrix into
    n_splits = int(n_features / block_size)
    X2 = X**2
    emp_cov_trace = tl.sum(X2) / n_samples
    mu = emp_cov_trace / n_features
    beta_ = tl.tensor(0.0)  # sum of the coefficients of <X2.T, X2>
    delta_ = tl.tensor(0.0)  # sum of the *squared* coefficients of <X.T, X>
    # starting block computation
    for i in range(n_splits):
        for j in range(n_splits):
            rows = slice(block_size * i, block_size * (i + 1))
            cols = slice(block_size * j, block_size * (j + 1))
            beta_ += tl.sum(tl.dot(X2.T[rows], X2[:, cols]))
            delta_ += tl.sum(tl.dot(X.T[rows], X[:, cols]) ** 2)
        rows = slice(block_size * i, block_size * (i + 1))
        beta_ += tl.sum(tl.dot(X2.T[rows], X2[:, block_size * n_splits :]))
        delta_ += tl.sum(tl.dot(X.T[rows], X[:, block_size * n_splits :]) ** 2)
    for j in range(n_splits):
        cols = slice(block_size * j, block_size * (j + 1))
        beta_ += tl.sum(tl.dot(X2.T[block_size * n_splits :], X2[:, cols]))
        delta_ += tl.sum(tl.dot(X.T[block_size * n_splits :], X[:, cols]) ** 2)
    delta_ += tl.sum(
        tl.dot(X.T[block_size * n_splits :], X[:, block_size * n_splits :]) ** 2
    )

    delta_ /= n_samples**2
    beta_ += tl.sum(
        tl.dot(X2.T[block_size * n_splits :], X2[:, block_size * n_splits :])
    )
    # use delta_ to compute beta

    beta = 1.0 / (n_features * n_samples) * (beta_ / n_samples - delta_)
    # delta is the sum of the squared coefficients of (<X.T,X> - mu*Id) / p
    delta = delta_ - 2.0 * mu * emp_cov_trace + n_features * mu**2
    delta /= n_features
    # get final beta as the min between beta and delta
    # We do this to prevent shrinking more than "1", which would invert
    # the value of covariances
    # beta = tl.min(tl.tensor([beta, delta]))
    # finally get shrinkage
    # shrinkage = 0 if beta == 0 else beta / delta
    shrinkage = beta / delta
    shrinkage = min(shrinkage, 1)
    shrinkage = max(shrinkage, 0)
    return shrinkage


def oas(emp_cov, n_samples):
    """Estimate covariance with the Oracle Approximating Shrinkage algorithm.

    The formulation is based on [1]_.
    [1] "Shrinkage algorithms for MMSE covariance estimation.",
        Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O.
        IEEE Transactions on Signal Processing, 58(10), 5016-5029, 2010.
        https://arxiv.org/pdf/0907.4698.pdf
    """
    n_features = emp_cov.shape[0]
    if n_features == 1:
        return 0

    p = n_features
    n = n_samples
    S = emp_cov

    num = (1 - 2 / p) * tl.trace(S**2) + tl.trace(S) ** 2
    den = (n + 1 - 2 / p) * (tl.trace(S**2) - tl.trace(S) ** 2 / n_features)
    shrinkage = num / den
    shrinkage = min(shrinkage, 1)
    shrinkage = max(shrinkage, 0)
    return shrinkage
