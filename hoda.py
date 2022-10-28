import warnings

import ipdb
import jax.scipy as scipy
import numpy as np
import scipy.linalg
import tensorly as tl
from sklearn.base import BaseEstimator, TransformerMixin
from tensorly import random as tl_random


class HODA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_iter=100,
        tol=1e-13,
        toeplitz=(1, 2),
        taper=None,
        rank=None,
        initialize="identity",
        shrinkage="lw",
        verbose=False,
        tl_context=None,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.toeplitz = toeplitz
        self.taper = taper
        self.rank = rank
        self.initialize = initialize
        self.shrinkage = shrinkage
        self.verbose = verbose
        self.tl_context = tl_context

    def fit(self, X, y):
        tl_context = self.tl_context
        if self.tl_context is None:
            tl_context = dict()
        X_orig = tl.tensor(X.copy(), **tl_context)
        X = tl.tensor(X, **tl_context)

        # Calculate class means
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        class_size = np.zeros(len(self.classes_), dtype=int)
        for c, cls in enumerate(self.classes_):
            class_size[c] = np.count_nonzero(y == cls)
        n_samples = X.shape[0]
        shape = X.shape[1:]
        order = len(shape)

        # Construct tapers
        tapers = [None] * order
        for k in range(order):
            if k == self.taper:
                taper = np.linspace(1, 0, num=shape[k])
                tapers[k] = tl.zeros((shape[k], shape[k]), **tl_context)
                for i in range(shape[k]):
                    taper_vec = np.repeat(taper[i], shape[k] - i)
                    tapers[k] += np.diag(taper_vec, i)
                    if i != 0:
                        tapers[k] += np.diag(taper_vec, -i)
            else:
                tapers[k] = tl.ones((shape[k], shape[k]), **tl_context)

        # Calculate means and center
        self.class_means_ = tl.zeros((n_classes, *shape), **tl_context)
        for i, cls in enumerate(self.classes_):
            self.class_means_ = self.class_means_.at[i].set(
                tl.mean(X[y == cls], axis=0)
            )
            X = X.at[y == cls].set(X[y == cls] - self.class_means_[i])
        self.mean_ = tl.mean(self.class_means_, axis=0)

        # Initialize projections
        self.projs_ = [None] * order
        for k in range(order):
            if self.initialize == "identity":
                self.projs_[k] = tl.eye(shape[k], self.rank[k], **tl_context)
            elif self.initialize == "random":
                self.projs_[k] = tl_random.random_tensor(
                    shape=(shape[k], self.rank[k]), **tl_context
                )
                self.projs_[k], _ = tl.qr(self.projs_[k], mode="reduced")
            elif self.initialize == "svd":
                Xt = tl.unfold(X, k + 1).T
                _, _, Vt = scipy.sparse.linalg.svds(
                    Xt, k=shape[k], return_singular_vectors="vh"
                )
                self.projs_[k] = Vt.T
            else:
                raise ValueError("initialize should be one of {identity, random, svd}")

        # Find projections
        self.updates_ = np.zeros((order, self.max_iter), dtype=float)
        self.objective_ = np.zeros((self.max_iter), dtype=float)
        self.scatter_w_ = [None] * order
        self.scatter_b_ = [None] * order
        for self.iter_ in range(self.max_iter):
            if self.verbose:
                print(f"[{self.iter_}/{self.max_iter}]", end="  ")
            new_projs = [None] * order
            for k in range(order):
                X_proj = self._project(X, k)

                # Calculate whithin class scatter

                X_proj = tl.base.unfold(X_proj, k + 1)
                scatter_w = X_proj @ X_proj.conj().T
                scatter_w /= tl.sum(tl.diag(scatter_w)) / shape[k]

                # modes = [k2 for k2 in range(order + 1) if k2 != k + 1]
                # scatter_w = tl.tenalg.tensordot(X_proj, X_proj.conj(), modes=modes)

                # Force toeplitz form
                if self.toeplitz is not None and k in self.toeplitz:
                    scatter_w = force_toeplitz(scatter_w)
                # Apply taper
                if self.taper is not None and self.toeplitz == k:
                    scatter_w *= tapers[k]
                # Normalize
                scatter_w /= tl.sum(tl.diag(scatter_w)) / shape[k]
                # Shrinkage regularization
                if self.shrinkage == "lw":
                    # shrinkage = schaefer_strimmer_shrinkage(
                    #    X_proj.T, scatter_w, n_samples
                    # )
                    shrinkage = tl_ledoit_wolf_shrinkage(
                        X_proj.T, n_samples, assume_centered=True
                    )
                else:
                    shrinkage = self.shrinkage[k]
                if self.verbose:
                    print(f"shrinkage[{k}]={shrinkage:.4f}", end="  ")
                mu = tl.sum(tl.diag(scatter_w)) / shape[k]
                scatter_w = (1 - shrinkage) * scatter_w + shrinkage * mu * tl.eye(
                    shape[k], **tl_context
                )
                # Force symmetry
                scatter_w = (scatter_w + scatter_w.conj().T) / 2

                self.scatter_w_[k] = scatter_w

                # Calculate between class scatter
                class_means_proj = self._project(self.class_means_, k)
                mean_proj = self._project(self.mean_[np.newaxis], k)
                class_means_proj -= mean_proj
                class_means_proj = tl.base.partial_unfold(
                    class_means_proj, k, skip_begin=1
                )
                scatter_b = tl.zeros((shape[k], shape[k]), **tl_context)
                for c in range(n_classes):
                    scatter_b += (
                        class_size[c]
                        * class_means_proj[c]
                        @ class_means_proj[c].conj().T
                    )
                scatter_b = (scatter_b + scatter_b.conj().T) / 2
                scatter_b /= tl.sum(tl.diag(scatter_b)) / shape[k]
                self.scatter_b_[k] = scatter_b

                # Solve
                eig_idc = [shape[k] - self.rank[k], shape[k] - 1]
                w, v = scipy.linalg.eigh(scatter_b, scatter_w, subset_by_index=eig_idc)
                v *= tl.sign(w)
                v = v[:, ::-1]
                # Orthonormalize for stability
                new_projs[k], _ = tl.qr(v, mode="reduced")

            # Stopping criterion
            break_flag = True
            for k in range(order):
                tol = self.tol * np.prod(self.projs_[k].shape)
                self.updates_[k, self.iter_] = tl.abs(
                    tl.norm(new_projs[k] - self.projs_[k])
                )
            if not self.updates_[k, self.iter_] < tol:
                break_flag = False
            self.projs_ = new_projs
            # self.objective_[self.iter_] = self.fisher_ratio(self.transform(X_orig), y)
            if self.verbose:
                print(f"step={np.mean(self.updates_[:, self.iter_]):.4e}", end="  ")
                print(f"objective={self.objective_[self.iter_]:.4e}")
            if break_flag:
                break
        return self

    def fisher_ratio(self, X, y, assume_centered=False, class_means=None):
        tl_context = self.tl_context
        if tl_context is None:
            tl_context = dict()
        classes = np.unique(y)
        n_classes = len(classes)
        class_size = np.zeros(len(self.classes_), dtype=int)
        for c, cls in enumerate(self.classes_):
            class_size[c] = np.count_nonzero(y == cls)
        class_size = tl.tensor(class_size, **tl_context)
        n_samples, n_features = X.shape
        class_means = tl.zeros((n_classes, n_features), **tl_context)
        for i, cls in enumerate(classes):
            class_means = class_means.at[i].set(tl.mean(X[y == cls], axis=0))
            X = X.at[y == cls].set(X[y == cls] - class_means[i][np.newaxis])
        mean = tl.mean(class_means, axis=0)
        scatter_w = tl.sum(tl.norm(X, axis=1))
        class_means -= mean[np.newaxis]
        scatter_b = tl.sum(tl.norm(class_means, axis=1) * class_size)
        return scatter_b / scatter_w

    def _project(self, X, k):

        order = len(X.shape) - 1
        return tl.tenalg.multi_mode_dot(
            X, self.projs_, modes=range(1, order + 1), skip=k, transpose=True
        )
        # X_proj = X
        # for k2 in range(order):
        #    if k2 != k:
        #        U = self.projs_[k2]
        #        X_proj = tl.tenalg.mode_dot(X_proj, U, mode=1 + k2, transpose=True)
        # return X_proj

    def transform(self, X, y=None):
        order = len(X.shape) - 1
        X_trans = tl.tenalg.multi_mode_dot(
            X, self.projs_, modes=range(1, order + 1), transpose=True
        )
        """
        order = len(X.shape) - 1
        X_trans = X.copy()
        for k in range(order):
            U = self.projs_[k]
            X_trans = tl.tenalg.mode_dot(X_trans, U, mode=1 + k, transpose=True)
        """
        X_trans = X_trans.reshape(X.shape[0], -1)
        return X_trans


def tl_toeplitz(toeplitz, r=None):
    # Form a 1-D array containing a reversed c followed by r[1:] that could be
    # strided to give us toeplitz matrix.
    n = len(toeplitz)
    toeplitz_full = tl.zeros((n, n), dtype=toeplitz.dtype)
    for i in range(n):
        for j in range(n):
            idx = abs(i - j)
            toeplitz_full = toeplitz_full.at[i, j].set(toeplitz[idx])
    return toeplitz_full


def force_toeplitz(cov):
    """Coerce the calculated empirical covariance to a Toeplitz-structured
    matrix by setting each diagonal to its mean value.
    """
    n_features = cov.shape[0]
    toeplitz = tl.zeros_like(cov[:, 0])
    for i in range(n_features):
        diag = tl.diag(cov, k=i)
        diag_mean = diag.mean()
        toeplitz = toeplitz.at[i].set(diag_mean)
    toeplitz = tl_toeplitz(toeplitz)

    return toeplitz


def schaefer_strimmer_shrinkage(X, cov, n_epochs):
    context = dict(device=X.device, dtype=X.dtype)
    n_features = X.sape[1]
    nu = tl.sum(tl.diag(cov)) / n_features
    num = 1
    raise NotImplementedError

    den = tl.sum(cov**2)
    den -= tl.sum(tl.diag(cov**2))
    den += tl.sum((tl.diag(cov) - nu) ** 2)
    shrinkage = (n_epochs / (n_epochs - 1) ** 2) * (num / den)
    return max(min(shrinkage, 1), 0)


def tl_ledoit_wolf_shrinkage(
    X,
    n_epochs,
    assume_centered=False,
    block_size=1000,
):
    # for only one feature, the result is the same whatever the shrinkage
    if len(X.shape) == 2 and X.shape[1] == 1:
        return 0.0
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))

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
    beta = min(beta, delta)
    # finally get shrinkage
    shrinkage = 0 if beta == 0 else beta / delta
    return shrinkage
