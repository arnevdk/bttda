# import numpy as np
import math
import warnings

import ipdb
import numpy as np
import tensorly as tl
from sklearn.base import BaseEstimator

import hoda.backend as backend

try:
    import cupy
except ImportError:
    pass


def norm_fro(A):
    return tl.sqrt(tl.sum(A**2))


def kron_pca(cov, n_components=2, outer=-1, inner=-1):
    cov_pvl = pvl_perm(cov, outer, inner)
    pass


def pvl_perm(cov, outer=-1, inner=-1):
    """Pitsianis-Van Loan permtation"""
    if outer == inner == -1:
        raise ValueError("Either outer or inner dimension be specified")
    total = cov.shape[0]
    if outer == -1:
        outer = total // inner
    if inner == -1:
        inner = total // outer
    cov_pvl = tl.zeros((outer * outer, inner * inner))
    for o1 in range(outer):
        for o2 in range(outer):
            block = cov[o1 * inner : (o1 + 1) * inner, o2 * inner : (o2 + 1) * inner]
            cov_pvl[o1 * outer + o2] = block.flatten()
    return cov_pvl


def pvl_perm(cov, outer=-1, inner=-1):
    """Pitsianis-Van Loan permtation"""
    if outer == inner == -1:
        raise ValueError("Either outer or inner dimension be specified")
    total = cov.shape[0]
    if outer == -1:
        outer = total // inner
    if inner == -1:
        inner = total // outer
    cov_pvl = tl.zeros((outer * outer, inner * inner))
    for o1 in range(outer):
        for o2 in range(outer):
            block = cov[o1 * inner : (o1 + 1) * inner, o2 * inner : (o2 + 1) * inner]
            cov_pvl[o1 * outer + o2] = block.flatten()
    return cov_pvl


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

        self.scale_ = (
            math.sqrt(norm_fro(X) * norm_fro(Y)) * math.prod(Y_shape) / (n_samples - 1)
        )

        # Initialize
        self.covs_ = [None] * order
        self.precs_ = [None] * order
        for k in range(order):
            self.covs_[k] = tl.zeros((X_shape[k], Y_shape[k]))
            self.covs_[k][: Y_shape[k], :] = tl.eye(Y_shape[k])
            self.precs_[k] = self.covs_[k].copy()

        if not self.assume_centered:
            _, X = center(X, y)
            _, Y = center(Y, y)

        for self.iter_ in range(1, self.max_iter + 1):
            new_covs = [None] * order
            update = 0
            if self.estimator == "mle":
                transforms = self.precs_
            elif self.estimator == "map":
                transforms = self.covs_
            else:
                raise NotImplementedError
            for k in range(order):
                modes = range(1, order + 1)
                X_proj = tl.tenalg.multi_mode_dot(
                    X, transforms, modes, skip=k, transpose=True
                )
                modes = [0] + [kk + 1 for kk in range(order) if kk != k]
                cov = tl.tensordot(X_proj, Y, axes=[modes, modes])
                cov /= norm_fro(cov)
                new_covs[k] = cov
                update += norm_fro(self.covs_[k] - cov) / self.scale_
            self.covs_ = new_covs
            for k in range(order):
                self.precs_[k] = backend.np.linalg.pinv(self.covs_[k]).T
            if update < self.tol:
                break
        return self


#
# class KroneckerSumCovariance(TensorCovariance):
#    def __init__(
#            self,
#            assume_centered=False,
#            n_components=2,
#            max_iter=64,
#            tol=1e-12,
#            verbose=False,
#            rewrite=False,
#            taper=True,
#    ):
#        super().__init__(assume_centered=assume_centered)
#        self.n_components = n_components
#        self.max_iter = max_iter
#        self.tol = tol
#        self.verbose = verbose
#        self.rewrite = rewrite
#        self.taper = taper
#
#    def fit(self, X, y=None):
#        self._covariance_ = None
#        self._sp_precs_ = None
#        self._tmp_precs_ = None
#        self._precision_ = None
#        X = self._center(X, y)
#        n_epochs, n_channels, n_samples = X.shape
#        self.n_components_ = min(n_channels - 1, n_samples - 1,
#                                 self.n_components)
#        XH = X.conj().transpose((0, 2, 1))
#        self.sp_covs_ = np.zeros((self.n_components_, n_channels, n_channels),
#                                 dtype=X.dtype)
#        self.tmp_covs_ = np.zeros((self.n_components_, n_samples, n_samples),
#                                  dtype=X.dtype)
#
#        # Initialize
#        self.sp_covs_[0] = np.eye(n_channels)
#        self.tmp_covs_[0] = np.eye(n_samples)
#        for c in range(1, self.n_components_):
#            self.sp_covs_[c] = np.eye(n_channels)
#            self.tmp_covs_[c] = np.eye(n_samples)
#        for c1 in range(self.n_components_):
#            if self.verbose:
#                print(f"Component [{c1 + 1}/{self.n_components_}]")
#            for iter in range(self.max_iter):
#                if self.verbose:
#                    print(f"Iteration {iter}", end="  ")
#                sp_cov = np.zeros((n_channels, n_channels), dtype=X.dtype)
#                tmp_cov = np.zeros((n_samples, n_samples), dtype=X.dtype)
#                for i in range(n_epochs):
#                    sp_cov += X[i] @ self.tmp_covs_[c1] @ XH[i]
#                    tmp_cov += XH[i] @ self.sp_covs_[c1] @ X[i]
#                    for c2 in range(c1):
#                        sp_cov -= np.trace(
#                            self.sp_covs_[c2] @ X[i] @ self.tmp_covs_[c1] \
#                            @ XH[i]) * self.sp_covs_[c2]
#                        tmp_cov -= np.trace(
#                            self.tmp_covs_[c2] @ XH[i] @ self.sp_covs_[c1] \
#                            @ X[i]) * self.tmp_covs_[c2] \
#                                   * scipy.linalg.norm(self.tmp_covs_[c2]) ** 2
#                sp_cov /= n_epochs * scipy.linalg.norm(self.tmp_covs_[c1]) ** 2
#                tmp_cov /= n_epochs
#
#                tmp_cov = force_toeplitz(tmp_cov)
#                if self.taper:
#                    taper = np.arange(n_samples)[::-1] / n_samples
#                    tmp_cov *= taper
#                tmp_cov = scipy.linalg.toeplitz(tmp_cov)
#
#                new_sp_cov_norm = sp_cov / scipy.linalg.norm(sp_cov)
#                old_sp_cov_norm = self.sp_covs_[c1] / scipy.linalg.norm(
#                    self.sp_covs_[c1])
#                new_tmp_cov_norm = tmp_cov / scipy.linalg.norm(tmp_cov)
#                old_tmp_cov_norm = self.tmp_covs_[c1] / scipy.linalg.norm(
#                    self.tmp_covs_[c1])
#                step = scipy.linalg.norm(new_sp_cov_norm - old_sp_cov_norm)
#                step *= scipy.linalg.norm(new_tmp_cov_norm - old_tmp_cov_norm)
#                step = np.sqrt(step)
#                if self.verbose:
#                    print(f"step: {step}")
#
#                self.sp_covs_[c1] = sp_cov
#                self.tmp_covs_[c1] = tmp_cov / scipy.linalg.norm(tmp_cov)
#                if step < self.tol:
#                    break
#
#        if self.rewrite:
#            for c in range(1, self.n_components_):
#                self._rewrite(c)
#
#        shape = (n_channels * n_samples, n_channels * n_samples)
#        self.covariance_ = np.zeros(shape)
#        for c in range(self.n_components_):
#            self.covariance_ += np.kron(self.sp_covs_[c],
#                                        self.tmp_covs_[c])
#        shrinkage = oas(self.covariance_, n_epochs)
#        self.covariance_ = shrunk_covariance(self.covariance_, shrinkage)
#        return self
#
#    def _rewrite(self, c1):
#        self._covariance_ = None
#        X1 = self.sp_covs_[c1 - 1]
#        X2 = self.sp_covs_[c1]
#        T1 = self.tmp_covs_[c1 - 1]
#        T2 = self.tmp_covs_[c1]
#
#        def construct_Xh_Th(abcd):
#            a = abcd[0]
#            b = abcd[1]
#            c = abcd[2]
#            d = (1 + b * c) / a
#
#            X1t = (X1 - c * X2) / a
#            X2t = (X1 - d * X2) / b
#            T1t = (T1 + b * T2) / d
#            T2t = (T2 + c * T1) / a
#
#            sX1h, UX1h = scipy.linalg.eigh(X1t, check_finite=False,
#                                           subset_by_value=(0, np.inf))
#            sX2h, UX2h = scipy.linalg.eigh(X2t, check_finite=False,
#                                           subset_by_value=(0, np.inf))
#            sT1h, UT1h = scipy.linalg.eigh(T1t, check_finite=False,
#                                           subset_by_value=(0, np.inf))
#            sT2h, UT2h = scipy.linalg.eigh(T2t, check_finite=False,
#                                           subset_by_value=(0, np.inf))
#
#            X1h = UX1h @ np.diag(sX1h) @ UX1h.conj().T
#            X2h = UX2h @ np.diag(sX2h) @ UX2h.conj().T
#            T1h = UT1h @ np.diag(sT1h) @ UT1h.conj().T
#            T2h = UT2h @ np.diag(sT2h) @ UT2h.conj().T
#
#            return X1h, X2h, T1h, T2h
#
#        full_cov = np.kron(X1, T1) + np.kron(X2, T2)
#
#        def cost(abcd):
#            X1h, X2h, T1h, T2h = construct_Xh_Th(abcd)
#            diff = full_cov - np.kron(X1h, T1h) - np.kron(X2h, T2h)
#            cost = scipy.linalg.norm(diff) ** 2
#            cost /= scipy.linalg.norm(full_cov) ** 2
#            return cost
#
#        abcd_init = np.random.rand(3)
#        solver_options = dict()
#        res = scipy.optimize.minimize(cost, x0=abcd_init,
#                                      options=solver_options)
#        if self.verbose:
#            print(res)
#        X1h, X2h, T1h, T2h = construct_Xh_Th(res.x)
#        self.sp_covs_[c1 - 1] = X1h
#        self.sp_covs_[c1] = X2h
#        self.tmp_covs_[c1 - 1] = T1h
#        self.tmp_covs_[c1] = T2h
#
#    @property
#    def sp_precs_(self):
#        if self._sp_precs_ is None:
#            self._sp_precs_ = np.zeros_like(self.sp_covs_)
#            for c in range(self.n_components_):
#                self._sp_precs_[c] = scipy.linalg.pinvh(self.sp_covs_[c])
#        return self._sp_precs_
#
#    @property
#    def tmp_precs_(self):
#        if self._tmp_precs_ is None:
#            self._tmp_precs_ = np.zeros_like(self.tmp_covs_)
#            for c in range(self.n_components_):
#                self._tmp_precs_[c] = scipy.linalg.pinvh(self.tmp_covs_[c])
#        return self._tmp_precs_
