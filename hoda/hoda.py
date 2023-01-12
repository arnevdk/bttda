import bisect
import math

import cupy
import ipdb
import numpy as np
import tensorly as tl
import tensorly.decomposition
import tensorly.tenalg
from kneed import KneeLocator
from sklearn.base import BaseEstimator, TransformerMixin
from tensorly import random as tl_random
from tqdm.notebook import tqdm

from hoda.tenalg import (det, force_toeplitz, lobpcg, pinvh, trunc_gevd,
                         trunc_svd)


def solve_ratio_svd(scatter_w, scatter_b, v, r):
    v, w = trunc_svd(pinvh(scatter_w) @ scatter_b, r)
    return v, w


def solve_ratio_gevd(scatter_w, scatter_b, v, r):
    v, w = trunc_gevd(scatter_b, scatter_w, r)
    return v, w


def solve_ratio_lanczos(scatter_w, scatter_b, v, r):
    raise NotImplementedError


def solve_ratio_lobpcg(scatter_w, scatter_b, v, r, **solver_params):
    solver_params["max_iter"] = 1
    w, v = lobpcg(scatter_b, v, B=scatter_w, largest=True, **solver_params)
    return v, w


def solve_diff_svd(scatter_w, scatter_b, v, r, psi=1):
    # phi = tl.trace(v.T @ scatter_b @ v) / tl.trace(v.T @ scatter_w @ v)
    _, psi = trunc_svd(pinvh(scatter_w) @ scatter_b, 1)
    v, w = trunc_svd(scatter_b - psi * scatter_w, r)
    return v, w


def solve_sr(scatter_w, scatter_b, v, w, r):
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
    X = X.copy()
    n_samples, *shape = X.shape
    classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(classes)

    class_means = tl.zeros((n_classes, *shape))
    for ci, c in enumerate(classes):
        class_means[ci] = tl.mean(X[y == c], axis=0)
        X[y == c] = X[y == c] - class_means[ci]
    class_means = tl.stack(class_means)
    class_mean = tl.mean(class_means, axis=0)

    scatter_b = 0
    for ci, c in enumerate(classes):
        scatter_b += (
            class_counts[ci] * tl.norm(class_means[ci] - class_mean, order=2) ** 2
        )
    scatter_w = tl.norm(X, order=2) ** 2
    return scatter_b / scatter_w


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
        self.keep_train_score = keep_train_score

    def fit(self, X, y):
        X = tl.tensor(X)
        self.classes_, class_counts = np.unique(y, return_counts=True)
        class_counts = tl.tensor(class_counts)
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
        class_means = tl.zeros((n_classes, *shape))
        X_centered = []
        class_mean = 0
        for ci, c in enumerate(self.classes_):
            where = y == c
            where = where.reshape((where.shape[0], 1, 1))
            X_where = X[y == c]
            class_means[ci] = tl.mean(X_where, axis=0)
            class_mean += class_means[ci] / n_classes
            X_centered.append(X_where - class_means[ci])
        X_centered = tl.concatenate(X_centered, axis=0)
        for ci, c in enumerate(self.classes_):
            class_means[ci] -= class_mean

        # Initialize rank
        self.rank_ = self.rank
        if self.rank_ is None:
            self.rank_ = list(shape)
            # core, factors = tl.decomposition.tucker(class_mean, rank=min(shape))
            # self.rank_ = [0] * order
            # for k in range(order):
            #    modes = tuple([kk for kk in range(order) if kk != k])
            #    mlsvs = tl.norm(core, order=2, axis=modes)
            #    kneedle = KneeLocator(
            #        np.arange(len(mlsvs)) + 1,
            #        cupy.asnumpy(mlsvs),
            #        curve="convex",
            #        direction="decreasing",
            #    )
            #    self.rank_[k] = kneedle.knee
            # self.rank_ = [0] * order
            # for k in range(order):
            #    scatter_w = self._scatter_w(X_centered, k)
            #    scatter_b = self._scatter_b(class_means, class_counts, k)
            #    v, w = SOLVERS[self.solver](scatter_w, scatter_b, None, min(shape))
            #    sum_w = tl.sum(w)
            #    last_aic = np.inf
            #    for r in range(1, min(shape) + 1):
            #        likelihood = tl.sum(w[:r]) / sum_w
            #        n_parameters = shape[k] * r - (r * (r + 1)) / 2
            #        aic = 2 * n_parameters / n_samples - 2 * np.log(likelihood)
            #        print(aic)
            #        if aic < last_aic:
            #            self.rank_[k] += 1
            #            last_aic = aic
            #        else:
            #            break
            #    print()
            # for k in range(order):
            #    self.rank_[k] = min(self.rank_)

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
        self.train_partial_score_ = tl.zeros((order, self.max_iter))
        self.scatter_w_ = [None] * order
        self.scatter_b_ = [None] * order

        iterator = range(self.max_iter)
        if self.verbose:
            iterator = tqdm(iterator)
        for self.iter_ in iterator:
            new_projs = [None] * order
            for k in range(order):
                # Calculate whithin class scatter
                modes = range(1, order + 1)
                X_proj = tl.tenalg.multi_mode_dot(
                    X_centered,
                    self.projs_,
                    modes=modes,
                    skip=k,
                    transpose=True,
                )
                scatter_w = self._scatter_w(X_proj, k)
                self.scatter_w_[k] = scatter_w

                # Calculate between class scatter
                class_means_proj = tl.tenalg.multi_mode_dot(
                    class_means, self.projs_, modes=modes, skip=k, transpose=True
                )
                scatter_b = self._scatter_b(class_means_proj, class_counts, k)
                self.scatter_b_[k] = scatter_b

                # Solve
                v, w = SOLVERS[self.solver](
                    scatter_w, scatter_b, self.projs_[k], shape[k], **solver_params
                )

                # Select rank
                self.rank_[k] = 0
                last_aic = np.inf
                for r in range(1, shape[k] + 1):
                    likelihood = tl.sum(w[:r]) / tl.sum(w)
                    n_parameters = shape[k] * r - (r * (r + 1)) / 2
                    aic = 2 * n_parameters / n_samples - 2 * np.log(likelihood)
                    if aic < last_aic:
                        self.rank_[k] += 1
                        last_aic = aic
                    else:
                        break
                v = v[:, : self.rank_[k]]
                w = w[: self.rank_[k]]

                # Calculate partial train score
                if self.keep_train_score:
                    self.train_partial_score_[k, self.iter_] = tl.trace(
                        v.T @ scatter_b @ v
                    ) / tl.trace(v.T @ scatter_w @ v)

                # Orthonormalize for stability
                v, _ = tl.qr(v, mode="reduced")
                new_projs[k] = v

            self.projs_ = new_projs
            # Calculate train score
            if self.keep_train_score:
                core = self._transform(X, y)
                self.train_score_[self.iter_] = fisher_score(core, y)
        if self.verbose:
            print(f"Fit tucker model of rank {self.rank_}")
        return self

    def _scatter_w(self, X, k):
        order = len(X.shape[1:])
        n_samples = X.shape[0]
        # Calculate scatter
        modes = [0] + [kk + 1 for kk in range(order) if kk != k]
        scatter_w = tl.tensordot(X, X, axes=(modes, modes))
        # Force symmetry
        scatter_w = (scatter_w + scatter_w.conj().T) / 2
        # Force toeplitz
        if self.toeplitz is not None and k in self.toeplitz:
            scatter_w = force_toeplitz(scatter_w, taper=True)
        # Shrink
        scatter_w, shrinkage = shrink(scatter_w, n_samples, self.shrinkage[k])
        return scatter_w

    def _scatter_b(self, class_means, class_counts, k):
        order = len(class_means.shape[1:])
        n_classes, *shape = class_means.shape
        # Calculate scatter
        modes = [kk for kk in range(order) if kk != k]
        scatter_b = 0
        for ci in range(n_classes):
            scatter_b += (
                tl.tensordot(class_means[ci], class_means[ci], axes=(modes, modes))
                * class_counts[ci]
            )
        # Force symmetry
        scatter_b = (scatter_b + scatter_b.conj().T) / 2
        return scatter_b

    def feature_score(self, X, y):
        n_classes = len(self.classes_)
        _, class_counts = np.unique(y, return_counts=True)

        core = self._transform(X, y)
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

    def _transform(self, X, y=None):
        order = len(X.shape) - 1
        Xt = tl.tenalg.multi_mode_dot(
            X, self.projs_, modes=range(1, order + 1), transpose=True
        )
        return Xt

    def _inv_transform(self, Xt, y=None):
        order = len(Xt.shape) - 1
        X = tl.tenalg.multi_mode_dot(
            Xt, self.projs_, modes=range(1, order + 1), transpose=False
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


class BTTDA(BaseEstimator, TransformerMixin):
    def __init__(
        self, block_rank=None, hoda_params=None, verbose=False, keep_train_score=False
    ):
        self.hoda_params = hoda_params
        self.block_rank = block_rank
        self.verbose = verbose
        self.keep_train_score = keep_train_score

    def fit(self, X, y):
        X = tl.tensor(X)
        X_orig = X.copy()

        self.classes_, class_counts = np.unique(y, return_counts=True)

        hoda_params = self.hoda_params
        if hoda_params is None:
            hoda_params = dict()

        self.blocks_ = []
        self.score_ = []
        for b in range(self.block_rank):
            if self.verbose:
                print(f"Fitting block {b+1}/{self.block_rank}...")
            # Fit Tucker block
            block = HODA(**hoda_params)
            block.fit(X, y)
            bisect.insort(self.blocks_, block, key=lambda x: -float(x.score(X_orig, y)))
            # Calculate model score
            self.score_.append(float(self.score(X_orig, y)))
            # Reconstruct and subtract for next iteration
            X_approx = block._inv_transform(block._transform(X))
            X = X - X_approx
            if self.verbose:
                print()

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

    def score(self, X, y):
        n_classes = len(self.classes_)
        _, class_counts = np.unique(y, return_counts=True)

        core = self._transform(X, y)
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


def shrink(cov, n_samples, shrinkage):
    n_features, _ = cov.shape
    # Shrinkage regularization
    if shrinkage == "oas":
        shrinkage = oas(cov, n_samples)
    mu = tl.sum(tl.diag(cov)) / n_features
    cov = cov + shrinkage * mu * tl.eye(n_features)
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
    shrinkage = max(min(shrinkage, 1), 0)
    return shrinkage
