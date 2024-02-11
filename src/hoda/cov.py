# import numpy as np
import warnings

import numpy as np
import tensorly as tl
from sklearn.base import BaseEstimator

from hoda.backend import toeplitz

try:
    import cupy
except ImportError:
    pass


def center(X, y, classes=None):
    _, *shape = X.shape
    order = len(shape)
    if classes is None:
        classes = np.unique(y)
    n_classes = len(classes)

    means = tl.zeros((n_classes, *shape), X.dtype)
    if tl.get_backend() == "cupy":
        X_centered = tl.zeros((n_classes, *X.shape))
        full_nan = cupy.full_like(X, cupy.nan)
        for ci, c in enumerate(classes):
            where = y == c
            where = cupy.array(where)
            where = np.expand_dims(where, axis=tuple(np.arange(1, order + 1)))
            X_where = cupy.where(
                where,
                X,
                full_nan,
            )
            means[ci] = cupy.nanmean(X_where, axis=0)
            X_centered[ci] = X_where - means[ci]
        X_centered = cupy.nansum(X_centered, axis=0)
    else:
        X_centered = []
        for ci, c in enumerate(classes):
            X_where = X[y == c]
            means[ci] = tl.mean(X_where, axis=0)
            X_centered.append(X_where - means[ci])
        X_centered = tl.concatenate(X_centered, axis=0)
    return means, X_centered


def mode_scatter(
    X, k, weights=None, shrinkage=0, toeplitz=None, taper=False, assume_centered=False
):
    """Calculate the scatter matrix along a given tensor mode"""
    _, *shape = X.shape
    order = len(shape)
    n_features = shape[k]
    if weights is not None:
        X = (X.T * weights).T
    # Determine mode scatter
    modes = [0] + [kk + 1 for kk in range(order) if kk != k]
    if not assume_centered:
        X = X - tl.mean(X, axis=0)
    scatter = tl.tensordot(X, X.conj(), axes=(modes, modes))
    # Force Toeplitz
    if toeplitz is not None and k in toeplitz:
        scatter = force_toeplitz(scatter, taper=taper)
    # Determine shrinkage
    if shrinkage == "lw":
        Xf = tl.unfold(X, k + 1).T
        shrinkage = ledoit_wolf_shrinkage(Xf, assume_centered=assume_centered)
    elif shrinkage == "oas":
        Xf = tl.unfold(X, k + 1).T
        shrinkage = oas(Xf, assume_centered=assume_centered, emp_cov=scatter)
    elif shrinkage == "ell1":
        raise NotImplementedError
    elif shrinkage == "ell2":
        raise NotImplementedError
    elif shrinkage == "ell3":
        raise NotImplementedError
    elif shrinkage == "schaefer":
        raise NotImplementedError
    elif shrinkage == "loocv":
        raise NotImplementedError
    # Shrink
    structured = (tl.trace(scatter) / n_features) * tl.eye(
        scatter.shape[0], dtype=X.dtype
    )
    scatter = (1 - shrinkage) * scatter + shrinkage * structured
    return scatter, shrinkage


def force_toeplitz(A, taper=False):
    n, _ = A.shape
    toep = tl.zeros(n, dtype=A.dtype)
    for i in range(n):
        diag = tl.diag(A, k=i)
        toep[i] = tl.mean(diag)
    if taper:
        taper = tl.arange(len(toep), 0, -1) - 1
        toep = toep * taper
    return toeplitz(toep)


def ledoit_wolf_shrinkage(X, assume_centered=False, block_size=1000):
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


def oas(X, emp_cov=None, assume_centered=False):
    """Estimate covariance with the Oracle Approximating Shrinkage algorithm.

    The formulation is based on [1]_.
    [1] "Shrinkage algorithms for MMSE covariance estimation.",
        Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O.
        IEEE Transactions on Signal Processing, 58(10), 5016-5029, 2010.
        https://arxiv.org/pdf/0907.4698.pdf
    """
    if len(X.shape) == 2 and X.shape[1] == 1:
        # for only one feature, the result is the same whatever the shrinkage
        if not assume_centered:
            X = X - X.mean()
        return np.atleast_2d((X**2).mean()), 0.0

    n_samples, n_features = X.shape

    if emp_cov is None:
        emp_cov = X.T @ X / (n_samples - 1)

    # The shrinkage is defined as:
    # shrinkage = min(
    # trace(S @ S.T) + trace(S)**2) / ((n + 1) (trace(S @ S.T) - trace(S)**2 / p), 1
    # )
    # where n and p are n_samples and n_features, respectively (cf. Eq. 23 in [1]).
    # The factor 2 / p is omitted since it does not impact the value of the estimator
    # for large p.

    # Instead of computing trace(S)**2, we can compute the average of the squared
    # elements of S that is equal to trace(S)**2 / p**2.
    # See the definition of the Frobenius norm:
    # https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
    alpha = tl.mean(emp_cov**2)
    mu = tl.trace(emp_cov) / n_features
    mu_squared = mu**2

    # The factor 1 / p**2 will cancel out since it is in both the numerator and
    # denominator
    num = alpha + mu_squared
    den = (n_samples + 1) * (alpha - mu_squared / n_features)
    # shrinkage = 1.0 if den == 0 else min(num / den, 1.0)
    shrinkage = num / den

    return shrinkage
