import warnings

import ipdb

try:
    import cupy.linalg
    import cupyx.scipy.linalg
    import cupyx.scipy.sparse.linalg
except ImportError:
    pass

import numpy as np
import scipy.linalg
import tensorly as tl


def pinvh(A):
    if tl.get_backend() == "cupy":
        return cupy.linalg.pinv(A)
    elif tl.get_backend() == "numpy":
        return scipy.linalg.pinvh(A)
    else:
        raise NotImplementedError


def solve(A, B):
    if tl.get_backend() == "cupy":
        return cupy.linalg.solve(A, B)
    elif tl.get_backend() == "numpy":
        return scipy.linalg.solve(A, B)
    else:
        raise NotImplementedError


def trunc_eigh(
    A, B=None, rank=None, largest=True, init=None, method="lanczos", solver_params=None
):
    """

    SVD eigensolver can only be used if  B^-1@A is semi-positive definite
    """
    if init is None:
        init = tl.eye(A.shape, dtype=A.dtype)
    init = init[:, :rank]
    solver_params = solver_params or dict()
    if method == "lanczos":
        v, w = lanczos(A, B=B, rank=rank, largest=largest, **solver_params)
    elif method == "svd":
        solver_params["flip_sign"] = True
        solver_params.setdefault("method", "truncated_svd")
        if largest:
            solver_params["n_eigenvecs"] = rank
        else:
            solver_params["n_eigenvecs"] = None
        if B is None:
            v, w, _ = tl.tenalg.svd_interface(A, **solver_params)
        else:
            v, w, _ = tl.tenalg.svd_interface(pinvh(B) @ A, **solver_params)
        if not largest:
            w = w[-rank:]
            v = v[:, -rank:]

    elif method == "lobpcg":
        v, w = lobpcg(A, init, rank=rank, largest=largest, B=B)
    return v, w


def lanczos(A, B=None, rank=None, largest=True, **kwargs):
    if rank is None:
        rank = A.shape[0]
    if tl.get_backend() == "cupy":
        if B is not None:
            M = pinvh(B) @ A
        else:
            M = A
        # if rank is None or rank == M.shape[0]:
        #    w, v = cupy.linalg.eigh(M)
        # else:
        #    which = "LA" if largest else "SA"
        #    w, v = cupyx.scipy.sparse.linalg.eigsh(
        #        M, k=rank, return_eigenvectors=True, which=which, **kwargs
        #    )
        w, v = cupy.linalg.eigh(M)
        if largest:
            w = w[-rank:]
            v = v[:, -rank:]
        else:
            w = w[:rank]
            v = v[:, :rank]
    elif tl.get_backend() == "numpy":
        if largest:
            n = A.shape[0]
            subset = [n - rank, n - 1]
        else:
            subset = [0, rank - 1]
        w, v = scipy.linalg.eigh(A, b=B, subset_by_index=subset, **kwargs)
    else:
        raise NotImplementedError
    return v, w


def lobpcg(A, init, B=None, rank=None, **kwargs):
    if tl.get_backend() == "cupy":
        w, v = cupyx.scipy.sparse.linalg.lobpcg(A, init, B=B, **kwargs)
    elif tl.get_backend() == "numpy":
        w, v = scipy.sparse.linalg.lobpcg(A, init, B=B, **kwargs)
    else:
        raise NotImplementedError
    return v, w


def toeplitz(a):
    if tl.get_backend() == "numpy":
        A = scipy.linalg.toeplitz(a)
    elif tl.get_backend() == "cupy":
        A = cupyx.scipy.linalg.toeplitz(a)
    else:
        raise NotImplementedError
    return A


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


def det(A):
    if tl.get_backend() == "numpy":
        return scipy.linalg.det(A)
    elif tl.get_backend() == "cupy":
        return cupy.linalg.det(A)
    else:
        raise NotImplementedError


def maximum(A, B):
    if tl.get_backend() == "numpy":
        return np.maximum(A, B)
    elif tl.get_backend() == "cupy":
        return cupy.maximum(A, B)
    else:
        raise NotImplementedError


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
    beta = tl.min(tl.tensor([beta, delta]))
    # finally get shrinkage
    # shrinkage = 0 if beta == 0 else beta / delta
    shrinkage = beta / delta
    return shrinkage
