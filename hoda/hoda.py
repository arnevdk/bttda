import math

import ipdb
import numpy as np
import tensorly as tl
import tensorly.decomposition
import tensorly.tenalg
from sklearn.base import BaseEstimator, TransformerMixin
from tensorly import random as tl_random
from tqdm.notebook import tqdm

from hoda.tenalg import (det, force_toeplitz, lobpcg, pinvh, trunc_gevd,
                         trunc_svd)


def solve_ratio_svd(scatter_w, scatter_b, r):
    v, w = trunc_svd(pinvh(scatter_w) @ scatter_b, r)
    return v, w


def solve_ratio_gevd(scatter_w, scatter_b, r):
    v, w = trunc_gevd(scatter_b, scatter_w, r)
    return v, w


def solve_ratio_lanczos(scatter_w, scatter_b, r):
    raise NotImplementedError


def solve_ratio_lobpcg(scatter_w, scatter_b, r, init=None, **solver_params):
    if init is None:
        init = tl.eye(scatter_w.shape[0])[:, :r]
    w, v = lobpcg(scatter_b, init, B=scatter_w, largest=True, **solver_params)
    return v, w


def solve_diff_svd(scatter_w, scatter_b, r, psi=1):
    phi = tl.trace(scatter_b) / tl.trace(scatter_w)
    v, w = trunc_svd(scatter_b - psi * phi * scatter_w, r)
    return v, w


def solve_sr(scatter_w, scatter_b, r):
    raise NotImplementedError


SOLVERS = dict(
    ratio_svd=solve_ratio_svd,
    ratio_gevd=solve_ratio_gevd,
    ratio_lanczos=solve_ratio_lanczos,
    ratio_lobpcg=solve_ratio_lobpcg,
    diff_svd=solve_diff_svd,
    sr=solve_sr,
)


class HODA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_iter=100,
        tol=1e-13,
        rank=None,
        init="mlsvd",
        shrinkage="oas",
        toeplitz=None,
        solver="ratio_gevd",
        verbose=False,
        solver_params=None,
        fisher_thresh=0.05,
        keep_train_score=False,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.rank = rank
        self.init = init
        self.shrinkage = shrinkage
        self.toeplitz = toeplitz
        self.solver = solver
        self.verbose = verbose
        self.solver_params = solver_params
        self.fisher_thresh = fisher_thresh
        self.keep_train_score = keep_train_score

    def fit(self, X, y):
        X = tl.tensor(X)
        self.classes_, class_counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        n_samples, *shape = X.shape
        order = len(shape)

        # Initialize solver
        if self.solver not in SOLVERS.keys():
            raise ValueError(f"solver must be one of {list(SOLVERS.keys())}")
        solver_params = self.solver_params
        if solver_params is None:
            solver_params = dict()

        # Calculate means and center
        class_means = []
        X_centered = []
        class_mean = 0
        for ci, c in enumerate(self.classes_):
            where = y == c
            where = where.reshape((where.shape[0], 1, 1))
            X_where = X[y == c]
            mean = tl.mean(X_where, axis=0)
            class_means += [mean]
            class_mean += mean / n_classes
            X_centered.append(X_where - mean)
        X_centered = tl.concatenate(X_centered, axis=0)
        for ci, c in enumerate(self.classes_):
            class_means[ci] -= class_mean

        self.rank_ = self.rank
        if self.rank_ is None:
            self.rank_ = shape

        # Initialize projections
        self.projs_ = [None] * order
        if self.init == "mlsvd":
            modes = tuple(range(1, order + 1))
            _, self.projs_ = tl.decomposition.partial_tucker(
                X_centered, modes, rank=self.rank_
            )
        else:
            for k in range(order):
                if self.init == "identity":
                    self.projs_[k] = tl.eye(shape[k], self.rank_[k])
                elif self.init == "ones":
                    self.projs_[k] = tl.ones((shape[k], self.rank_[k]))
                elif self.init == "random":
                    self.projs_[k] = tl_random.random_tensor(
                        shape=(shape[k], self.rank_[k]),
                    )
                    self.projs_[k], _ = tl.qr(self.projs_[k], mode="reduced")
                elif self.init == "svd":
                    x = tl.unfold(X, k + 1)
                    self.projs_[k], _ = trunc_svd(x, self.rank_[k])
                else:
                    raise ValueError(
                        "init should be one of {identity, ones, random, svd}"
                    )

        # Find projections
        self.train_score_ = tl.zeros(self.max_iter)
        self.scatter_w_ = [None] * order
        self.scatter_b_ = [None] * order
        self.feature_mask_ = tl.ones(self.rank_, dtype=int)

        iterator = range(self.max_iter)
        if self.verbose:
            iterator = tqdm(iterator)
        for self.iter_ in iterator:
            new_projs = [None] * order
            for k in range(order):
                # Calculate whithin class scatter
                modes = range(1, order + 1)
                X_proj = tl.tenalg.multi_mode_dot(
                    X_centered, self.projs_, modes=modes, skip=k, transpose=True
                )
                modes = [0] + [kk + 1 for kk in range(order) if kk != k]
                scatter_w = tl.tensordot(X_proj, X_proj, axes=(modes, modes))
                # Force symmetry
                scatter_w = (scatter_w + scatter_w.conj().T) / 2
                # Force toeplitz
                if self.toeplitz is not None and k in self.toeplitz:
                    scatter_w = force_toeplitz(scatter_w, taper=True)
                # Shrink
                scatter_w, shrinkage = shrink(scatter_w, n_samples, self.shrinkage[k])
                self.scatter_w_[k] = scatter_w

                # Calculate between class scatter
                scatter_b = 0
                modes = range(order)
                for c in range(n_classes):
                    class_mean_proj = tl.tenalg.multi_mode_dot(
                        class_means[c], self.projs_, modes=modes, skip=k, transpose=True
                    )
                    class_mean_proj = tl.base.unfold(class_mean_proj, mode=k)
                    scatter_b += (
                        class_mean_proj @ class_mean_proj.conj().T
                    ) * class_counts[c]

                # Force symmetry
                scatter_b = (scatter_b + scatter_b.conj().T) / 2
                self.scatter_b_[k] = scatter_b

                # Solve
                if self.solver == "ratio-lobpcg":
                    solver_params["init"] = self.projs_[k]
                    solver_params["maxiter"] = 1
                v, w = SOLVERS[self.solver](
                    scatter_w, scatter_b, self.rank_[k], **solver_params
                )
                # Orthonormalize for stability
                v, _ = tl.qr(v, mode="reduced")
                new_projs[k] = v

            self.projs_ = new_projs

            # Calculate train score
            if self.keep_train_score:
                self.train_score_[self.iter_] = self.score(X, y)

        # Calculate feature mask
        # feature_score = self.feature_score(X, y)
        # self.feature_mask_[feature_score < self.fisher_thresh] = 0

        if self.verbose:
            print(f"Fit tucker model of rank {self.rank_}")
        return self

    def score(self, X, y):
        n_classes = len(self.classes_)
        _, class_counts = np.unique(y, return_counts=True)

        core = self.core(X, y)
        n_samples, *core_shape = core.shape
        class_means = tl.zeros((n_classes, *core_shape))
        for ci, c in enumerate(self.classes_):
            class_means[ci] = tl.mean(core[y == c], axis=0)
            core[y == c] = core[y == c] - class_means[ci]
        class_means = tl.stack(class_means)
        class_mean = tl.mean(class_means, axis=0)

        scatter_b = 0
        for ci, c in enumerate(self.classes_):
            scatter_b += (
                class_counts[ci] * tl.norm(class_means[ci] - class_mean, order=2) ** 2
            )
        scatter_w = tl.norm(core, order=2) ** 2
        return scatter_b / scatter_w

    def feature_score(self, X, y):
        n_classes = len(self.classes_)
        _, class_counts = np.unique(y, return_counts=True)

        core = self.core(X, y)
        core_flat = tl.unfold(core, 0)
        n_features = core_flat.shape[-1]
        fisher_score = tl.zeros(self.rank_)
        for f in range(n_features):
            g = core_flat[:, f]
            class_means = tl.zeros(n_classes)
            for ci, c in enumerate(self.classes_):
                class_means[ci] = tl.mean(g[y == c])
                g[y == c] = g[y == c] - class_means[ci]
            class_mean = tl.mean(tl.tensor(class_means))
            var_b = 0
            for ci, c in enumerate(self.classes_):
                var_b += class_counts[ci] * (class_means[ci] - class_mean) ** 2
            var_w = tl.sum(g**2)
            idx = np.unravel_index(f, self.rank_)
            fisher_score[*idx] = var_b / var_w
        return fisher_score

    def core(self, X, y=None):
        X = tl.tensor(X)
        order = len(X.shape) - 1
        Xt = tl.tenalg.multi_mode_dot(
            X, self.projs_, modes=range(1, order + 1), transpose=True
        )
        Xt *= self.feature_mask_
        return Xt

    def transform(self, X, y=None):
        Xt = self.core(X, y)
        return tl.to_numpy(Xt)

    def inv_transform(self, Xt, y=None):
        Xt = tl.tensor(Xt)
        order = len(Xt.shape) - 1
        X = tl.tenalg.multi_mode_dot(
            Xt, self.projs_, modes=range(1, order + 1), transpose=False
        )
        return X


class BTTDA(BaseEstimator, TransformerMixin):
    def __init__(self, block_rank=None, hoda_params=None, verbose=False):
        self.hoda_params = hoda_params
        self.block_rank = block_rank
        self.verbose = verbose

    def fit(self, X, y):
        X = X.copy()
        hoda_params = self.hoda_params
        if hoda_params is None:
            hoda_params = dict()

        self.blocks_ = []
        for b in range(self.block_rank):
            if self.verbose:
                print(f"Fitting block {b+1}/{self.block_rank}...")
            block = HODA(**hoda_params)
            block.fit(X, y)
            self.blocks_.append(block)
            X_approx = block.inv_transform(block.transform(X))
            X -= X_approx
            if self.verbose:
                print()

    def transform(self, X, y=None):
        Xt = []
        for block in self.blocks_:
            Xt.append(block.transform(X, y))
        Xt = np.stack(Xt, axis=1)
        return Xt


def shrink(cov, n_samples, method):
    n_features, _ = cov.shape
    # Shrinkage regularization
    if method == "oas":
        shrinkage = oas(cov, n_samples)
    mu = tl.sum(tl.diag(cov)) / n_features
    cov = (1 - shrinkage) * cov + shrinkage * mu * tl.eye(n_features)
    return cov, shrinkage


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
    return max(min(shrinkage, 1), 0)
