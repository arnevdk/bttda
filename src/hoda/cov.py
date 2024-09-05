# import numpy as np
import math
import pdb
import warnings

import numpy as np
import tensorly as tl
from sklearn.base import BaseEstimator

from hoda.backend import toeplitz, var

try:
    import cupy
except ImportError:
    pass

eyes = dict()


def mode_scatter(
    X, k, weights=None, shrinkage=0, toeplitz=None, taper=False, assume_centered=False
):
    """Calculate the scatter matrix along a given tensor mode"""
    global eyes

    n_samples, *shape = X.shape
    order = len(shape)
    n_features = shape[k]
    if weights is not None:
        X = (X.T * weights).T
    # Determine mode scatter
    modes = [0] + [kk + 1 for kk in range(order) if kk != k]
    if not assume_centered:
        X = X - tl.mean(X, axis=0)
    scatter = tl.tensordot(X, X, axes=(modes, modes))
    # Xf = tl.unfold(X, k + 1)
    # scatter = Xf @ Xf.T
    # Force Toeplitz
    if toeplitz is not None and k in toeplitz:
        scatter = force_toeplitz(scatter, taper=taper)
    # Determine shrinkage
    if shrinkage == "lw":
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
    # Shrink
    # if not n_features in eyes.keys():
    #    eyes[n_features] = tl.eye(n_features)
    # structured = eyes[n_features]
    structured = tl.eye(n_features)
    structured *= tl.trace(scatter) / n_features
    scatter = (1 - shrinkage) * scatter + shrinkage * structured
    return scatter, shrinkage


def force_toeplitz(A, taper=False):
    n, _ = A.shape
    toep = tl.zeros(n)
    for i in range(n):
        diag = tl.diag(A, k=i)
        toep[i] = tl.mean(diag)
    if taper:
        taper = tl.arange(len(toep), 0, -1) - 1
        toep = toep * taper
    return toeplitz(toep)


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
    emp_cov_trace = tl.sum(X2, axis=0) / n_samples
    mu = tl.sum(emp_cov_trace) / n_features
    beta_ = 0.0  # sum of the coefficients of <X2.T, X2>
    delta_ = 0.0  # sum of the *squared* coefficients of <X.T, X>
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
    delta = delta_ - 2.0 * mu * emp_cov_trace.sum() + n_features * mu**2
    delta /= n_features
    # get final beta as the min between beta and delta
    # We do this to prevent shrinking more than "1", which would invert
    # the value of covariances
    # beta = tl.min(tl.tensor([beta, delta]))
    # finally get shrinkage
    # shrinkage = 0 if beta == 0 else beta / delta
    shrinkage = beta / delta
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


def pvl_perm(X, n, m):
    Xs = np.zeros((n**2, m**2))
    for i in range(n):
        for j in range(n):
            block_ij = X[i * m : (i + 1) * m, j * m : (j + 1) * m]
            Xs[i * n + j, :] = block_ij.flatten()
    return Xs


def pvl_perm_inv(Xs, n, m):
    X = np.zeros((n * m, n * m))
    for i in range(n):
        for j in range(n):
            row_ij = Xs[i * n + j, :]
            X[i * m : (i + 1) * m, j * m : (j + 1) * m] = row_ij.reshape((m, m))
    return X


class KroneckerCovariance(BaseEstimator):
    def __init__(self, max_iter=1000, tol=1e-8, assume_centered=False, estimator="mle"):
        self.max_iter = max_iter
        self.tol = tol
        self.assume_centered = assume_centered
        self.estimator = estimator

    def fit(self, X, Y=None, y=None):
        if X is None:
            X = Y
        n_samples, *X_shape = X.shape
        _, *Y_shape = Y.shape
        order = len(X.shape[1:])

        # Initialize
        transforms = [None] * order
        covs = [None] * order
        for k in range(order):
            transforms[k] = tl.zeros((X_shape[k], Y_shape[k]))
            transforms[k][: Y_shape[k], :] = tl.eye(Y_shape[k])
            covs[k] = transforms[k].copy()

        # Find transformations
        for self.iter_ in range(1, self.max_iter + 1):
            update = 0
            for k in range(order):
                modes = range(1, order + 1)
                X_proj = tl.tenalg.multi_mode_dot(
                    X, transforms, modes, skip=k, transpose=True
                )
                modes = [0] + [kk + 1 for kk in range(order) if kk != k]
                cov = tl.tensordot(X_proj, Y, axes=[modes, modes])
                cov /= tl.mean(tl.diag(cov))
                update += tl.norm(covs[k] - cov)
                covs[k] = cov
                if self.estimator == "mle":
                    transforms[k] = cupy.linalg.pinv(cov).T
                elif self.estimator == "map":
                    transforms[k] = cov
                else:
                    raise NotImplementedError
            if update < self.tol:
                break

        self.covs_ = [None] * order
        # Find scaled covariance
        for k in range(order):
            modes = range(1, order + 1)
            X_proj = tl.tenalg.multi_mode_dot(
                X, transforms, modes, skip=k, transpose=True
            )
            modes = [0] + [kk + 1 for kk in range(order) if kk != k]
            self.covs_[k] = tl.tensordot(X_proj, Y, axes=[modes, modes])
            self.covs_[k] /= n_samples * math.prod(X_shape) / X_shape[k] - 1

        return self
