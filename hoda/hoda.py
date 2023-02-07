import bisect

import cupy
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import tensorly as tl
import tensorly.decomposition
import tensorly.tenalg
from kneed import KneeLocator
from sklearn.base import BaseEstimator, TransformerMixin
from tensorly import random as tl_random
from tqdm.notebook import tqdm

from hoda.tenalg import (det, force_toeplitz, lobpcg, pinvh, trunc_gevd,
                         trunc_svd)


def solve_ratio_svd(scatter_b, scatter_t, v, r):
    v, w, _ = trunc_svd(pinvh(scatter_t) @ scatter_b, r)
    return v, w


def solve_ratio_gevd(scatter_b, scatter_t, v, r):
    v, w = trunc_gevd(scatter_b, scatter_t, r)
    return v, w


def solve_ratio_lanczos(scatter_b, scatter_t, v, r):
    raise NotImplementedError


def solve_ratio_lobpcg(scatter_b, scatter_t, v, r, **solver_params):
    solver_params["max_iter"] = 1
    w, v = lobpcg(scatter_b, v, B=scatter_t, largest=True, **solver_params)
    return v, w


def solve_diff_svd(scatter_b, scatter_t, v, r, psi=1):
    psi = tl.trace(v.T @ scatter_b @ v) / tl.trace(v.T @ scatter_t @ v)
    v, w, _ = trunc_svd(scatter_b - psi * scatter_t, r)
    return v, w


def solve_sr(scatter_w, scatter_t, v, w, r):
    raise NotImplementedError


SOLVERS = dict(
    ratio_svd=solve_ratio_svd,
    ratio_gevd=solve_ratio_gevd,
    ratio_lanczos=solve_ratio_lanczos,
    ratio_lobpcg=solve_ratio_lobpcg,
    diff_svd=solve_diff_svd,
    sr=solve_sr,
)


def fisher_score(X, y):
    X = tl.tensor(X)
    n_samples, *shape = X.shape
    classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(classes)

    # Calculate class means, overall class mean and center data
    means = tl.zeros((n_classes, *shape))
    for ci, c in enumerate(classes):
        means[ci] = tl.mean(X[y == c], axis=0)
        X[y == c] -= means[ci][np.newaxis]
    means = tl.stack(means)
    class_mean = tl.mean(means, axis=0)

    # Calculate between class scatter
    scatter_b = 0
    for ci, c in enumerate(classes):
        scatter_b += class_counts[ci] * tl.norm(means[ci] - class_mean, order=2) ** 2
    # Calculate whithin class scatter
    scatter_w = tl.norm(X, order=2) ** 2
    # Calculate total scatter
    scatter_t = scatter_b + scatter_w

    return scatter_b / scatter_t


class HODA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_iter=100,
        tol=1e-12,
        rank=None,
        explain=0.99,
        init="mlsvd",
        shrinkage="oas",
        toeplitz=None,
        solver="ratio_gevd",
        verbose=False,
        solver_params=None,
        keep_train_info=False,
        priors=None,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.rank = rank
        self.explain = explain
        self.init = init
        self.shrinkage = shrinkage
        self.toeplitz = toeplitz
        self.solver = solver
        self.verbose = verbose
        self.solver_params = solver_params
        self.keep_train_info = keep_train_info
        self.priors = priors

    def fit(self, X, y):
        X = tl.tensor(X)
        self.classes_, class_counts = np.unique(y, return_counts=True)
        class_counts = tl.tensor(class_counts)
        n_classes = len(self.classes_)
        n_samples, *shape = X.shape
        order = len(shape)

        # Calculate priors
        if self.priors is None:  # estimate priors from sampl
            self.priors_ = class_counts / n_samples
        else:
            self.priors_ = np.asarray(self.priors)

        # Initialize solver.T
        if self.solver not in SOLVERS.keys():
            raise ValueError(f"solver must be one of {list(SOLVERS.keys())}")
        solver_params = self.solver_params
        if solver_params is None:
            solver_params = dict()

        # Calculate mode total scatter
        scatter_t_orig = [None] * order
        for k in range(order):
            modes = [0] + [kk + 1 for kk in range(order) if kk != k]
            scatter_t_orig[k] = tl.tensordot(X, X, axes=(modes, modes))

        # Calculate means and center
        means = tl.zeros((n_classes, *shape))
        X_centered = []
        class_mean = 0
        for ci, c in enumerate(self.classes_):
            where = y == c
            where = where.reshape((where.shape[0], 1, 1))
            X_where = X[y == c]
            means[ci] = tl.mean(X_where, axis=0)
            class_mean += means[ci] / n_classes
            X_centered.append(X_where - means[ci])
        X_centered = tl.concatenate(X_centered, axis=0)
        self.means_ = means.copy()
        for ci, c in enumerate(self.classes_):
            means[ci] -= class_mean

        # Initialize rank
        if self.rank is None:
            self.rank_ = [0] * order
            for k in range(order):
                # v, w, _ = trunc_svd(scatter_t_orig[k])
                scatter_w = self._scatter_w(X, k)
                scatter_b = self._scatter_b(means, class_counts, k)
                scatter_t = scatter_b + scatter_w
                v, w = SOLVERS[self.solver](
                    scatter_b,
                    scatter_t,
                    None,
                    None,
                    **solver_params,
                )
                sum_w = tl.sum(w)
                for r in range(1, shape[k] + 1):
                    if tl.sum(w[:r]) / sum_w >= self.explain:
                        self.rank_[k] = r
                        break
            for k in range(order):
                self.rank_[k] = min(self.rank_)
        else:
            self.rank_ = self.rank

        # Initialize projections
        self.scalings_ = [None] * order
        if self.init == "mlsvd":
            modes = tuple(range(1, order + 1))
            _, self.scalings_ = tl.decomposition.partial_tucker(
                X_centered, modes, rank=self.rank_
            )
        else:
            for k in range(order):
                if self.init == "identity":
                    self.scalings_[k] = tl.eye(shape[k], self.rank_[k])
                elif self.init == "ones":
                    self.scalings_[k] = tl.ones((shape[k], self.rank_[k]))
                elif self.init == "random":
                    self.scalings_[k] = tl_random.random_tensor(
                        shape=(shape[k], self.rank_[k]),
                    )
                    self.scalings_[k], _ = tl.qr(self.scalings_[k], mode="reduced")
                elif self.init == "svd":
                    x = tl.unfold(X, k + 1)
                    self.scalings_[k], _ = trunc_svd(x, self.rank_[k])
                else:
                    raise ValueError(
                        "init should be one of {identity, ones, random, svd}"
                    )

        # Store initial training information
        if self.keep_train_info:
            core = self._transform(X, y)
            f_score = fisher_score(core, y)
            mse = tl.mean((self._inv_transform(core) - X) ** 2)
            # TODO: mode f score
            self.train_info_ = dict(
                f_score=[float(f_score)],
                mode_f_score=[[] for _ in range(order)],
                mse=[float(mse)],
                update=[],
                mode_update=[[] for _ in range(order)],
            )

        # Iteratively finde projections
        self.scatter_w_ = [None] * order
        self.scatter_b_ = [None] * order
        last_scalings = self.scalings_
        iterator = range(self.max_iter)
        if self.verbose:
            iterator = tqdm(iterator)
        for self.iter_ in iterator:
            new_scalings = [None] * order
            for k in range(order):
                # Calculate whithin class scatter
                modes = range(1, order + 1)
                X_proj = tl.tenalg.multi_mode_dot(
                    X_centered,
                    self.scalings_,
                    modes=modes,
                    skip=k,
                    transpose=True,
                )
                scatter_w = self._scatter_w(X_proj, k)
                self.scatter_w_[k] = scatter_w

                # Calculate between class scatter
                means_proj = tl.tenalg.multi_mode_dot(
                    means, self.scalings_, modes=modes, skip=k, transpose=True
                )
                scatter_b = self._scatter_b(means_proj, class_counts, k)
                self.scatter_b_[k] = scatter_b

                # Solve
                scatter_t = scatter_b + scatter_w
                v, w = SOLVERS[self.solver](
                    scatter_b,
                    scatter_t,
                    self.scalings_[k],
                    self.rank_[k],
                    **solver_params,
                )
                # rtol = 1e-12
                # v = v[:, w / tl.sum(w) > rtol]
                # w = w[w / tl.sum(w) > rtol]
                # self.rank_[k] = v.shape[1]
                v, w, _ = trunc_svd(
                    v @ v.T @ scatter_t_orig[k] @ v @ v.T, self.rank_[k]
                )
                v, _ = tl.qr(v, mode="reduced")
                new_scalings[k] = v

            self.scalings_ = new_scalings

            # Calculate update
            self.scalings_ = new_scalings

            mode_update = tl.zeros(order)
            for k in range(order):
                last_r = last_scalings[k].shape[1]
                new_r = new_scalings[k].shape[1]
                max_r = max(last_r, new_r)
                last_proj_ext = tl.zeros((shape[k], max_r))
                last_proj_ext[:, :last_r] = last_scalings[k]
                new_proj_ext = tl.zeros((shape[k], max_r))
                new_proj_ext[:, :new_r] = new_scalings[k]
                mode_update[k] = tl.norm(
                    new_proj_ext.T @ last_proj_ext - tl.eye(max_r), order=2
                )
            update = tl.sum(mode_update)

            # Store training information
            if self.keep_train_info:
                core = self._transform(X, y)
                f_score = fisher_score(core, y)
                mse = tl.mean((self._inv_transform(core) - X) ** 2)
                self.train_info_["f_score"].append(float(f_score))
                self.train_info_["mse"].append(float(mse))
                self.train_info_["update"].append(float(update))
                for k in range(order):
                    self.train_info_["mode_update"][k].append(float(mode_update[k]))

            # Check convergence
            if update < self.tol:
                break
            last_scalings = self.scalings_

        # Calculate coefficients and intercept
        self.coef_ = tl.tenalg.multi_mode_dot(
            self.means_, self.scalings_, modes=(1, 2), transpose=True
        )
        self.coef_ = tl.tenalg.multi_mode_dot(
            self.coef_, self.scalings_, modes=(1, 2), transpose=False
        )

        means_flat = self.means_.reshape((n_classes, -1))
        coef_flat = self.means_.reshape((n_classes, -1))
        self.intercept_ = -0.5 * np.diag(np.dot(means_flat, coef_flat.T)) + np.log(
            self.priors_
        )

        if n_classes == 2:
            self.coef_ = self.coef_[1] - self.coef_[0]
            self.intercept_ = self.intercept_[1] - self.intercept_[0]

        if self.verbose:
            print(f"Fit tucker model of rank {self.rank_}")
        return self

    def _scatter_w(self, X, k):
        # order = len(X.shape[1:])
        # n_samples = X.shape[0]
        ## Calculate scatter
        # modes = [0] + [kk + 1 for kk in range(order) if kk != k]
        # scatter_w = tl.tensordot(X, X, axes=(modes, modes))
        # scatter_w, shrinkage = shrink(scatter_w, n_samples, self.shrinkage[k])

        # Shrink
        X_mode = tl.unfold(X, k + 1)
        scatter_w, shrinkage = ledoit_wolf(X_mode)
        # Force Toeplitz structure
        if self.toeplitz is not None and k in self.toeplitz:
            scatter_w = force_toeplitz(scatter_w, taper=False)
        return scatter_w

    def _scatter_b(self, means, class_counts, k):
        order = len(means.shape[1:])
        n_classes, *shape = means.shape
        # Calculate scatter
        modes = [kk for kk in range(order) if kk != k]
        scatter_b = 0
        for ci in range(n_classes):
            scatter_b += (
                tl.tensordot(means[ci], means[ci], axes=(modes, modes))
                * class_counts[ci]
            )
        # Force symmetry
        scatter_b = (scatter_b + scatter_b.conj().T) / 2
        return scatter_b

    def _transform(self, X, y=None):
        order = len(X.shape) - 1
        Xt = tl.tenalg.multi_mode_dot(
            X, self.scalings_, modes=range(1, order + 1), transpose=True
        )
        return Xt

    def _inv_transform(self, Xt, y=None):
        order = len(Xt.shape) - 1
        X = tl.tenalg.multi_mode_dot(
            Xt, self.scalings_, modes=range(1, order + 1), transpose=False
        )
        return X

    def transform(self, X, y=None):
        X = tl.tensor(X)
        Xt = self._transform(X, y)
        return tl.to_numpy(Xt)

    def inv_transform(self, Xt, y=None):
        Xt = tl.tensor(Xt)
        X = self._inv_transform(Xt, y)
        return tl.to_numpy(X)

    def decision_function(self, X):
        pass

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError


class BTTDA(BaseEstimator, TransformerMixin):
    def __init__(
        self, n_blocks=None, hoda_params=None, verbose=False, keep_train_score=False
    ):
        self.hoda_params = hoda_params
        self.n_blocks = n_blocks
        self.verbose = verbose
        self.keep_train_score = keep_train_score

    def fit(self, X, y):
        X = tl.tensor(X.copy())
        X_orig = X.copy()
        self.classes_, class_counts = np.unique(y, return_counts=True)

        hoda_params = self.hoda_params
        if hoda_params is None:
            hoda_params = dict()

        self.blocks_ = []
        self.train_mse_ = []
        X_rec = 0
        for b in range(self.n_blocks):
            if self.verbose:
                print(f"Fitting block {b+1}/{self.n_blocks}...")
            # Fit Tucker block
            block = HODA(**hoda_params)
            block.fit(X, y)
            self.blocks_.append(block)
            # Reconstruct and subtract for next iteration
            X_approx = block._inv_transform(block._transform(X))
            X -= X_approx
            if self.verbose:
                print()
            X_rec += X_approx
            mse = tl.mean((X_orig - X_rec) ** 2)
            self.train_mse_.append(float(mse))

        return self

    def _transform(self, X, y=None):
        n_samples, *_ = X.shape
        Xt = []
        for block in self.blocks_:
            Xtb = block._transform(X, y)
            Xt.append(Xtb.reshape(n_samples, -1))
        Xt = tl.concatenate(Xt, axis=1)
        return Xt

    def transform(self, X, y=None):
        X = tl.tensor(X)
        Xt = self._transform(X, y=None)
        Xt = tl.to_numpy(Xt)
        return Xt


def shrink(cov, n_samples, shrinkage):
    n_features, _ = cov.shape
    # Shrinkage regularization
    if shrinkage == "oas":
        shrinkage = oas(cov, n_samples)
    mu = tl.sum(tl.diag(cov)) / n_features
    cov = cov + shrinkage * mu * tl.eye(n_features)
    return cov, shrinkage


def ledoit_wolf(X, gamma=None, T=None, S=None):
    p, n = X.shape
    # Xn = X - np.repeat(np.mean(X, axis=1, keepdims=True), n, axis=1)
    Xn = X.copy()
    if S is None:
        S = np.matmul(Xn, Xn.T)
    Xn2 = Xn**2
    idxdiag = np.diag_indices(p)

    nu = tl.mean(S[idxdiag])
    if T is None:
        T = nu * tl.eye(p, p)

    # Ledoit Wolf
    V = 1.0 / (n - 1) * (Xn2 @ Xn2.T - S**2 / n)
    if gamma is None:
        gamma = n * tl.sum(V) / tl.sum((S - T) ** 2)
    if gamma > 1:
        print("logger.warning('forcing gamma to 1')")
        gamma = 1
    elif gamma < 0:
        print("logger.warning('forcing gamma to 0')")
        gamma = 0
    Cstar = (gamma * T + (1 - gamma) * S) / (n - 1)

    return Cstar, gamma


def oas(emp_cov, n_samples):
    n_features = emp_cov.shape[0]
    mu = tl.trace(emp_cov) / n_features

    # formula from Chen et al.'s **implementation**
    alpha = tl.mean(emp_cov**2)
    num = alpha + mu**2
    den = (n_samples + 1.0) * (alpha - (mu**2) / n_features)
    if den == 0:
        shrinkage = 1
    else:
        shrinkage = num / den
    shrinkage = max(min(shrinkage, 1), 0)
    return shrinkage
    return shrinkage
