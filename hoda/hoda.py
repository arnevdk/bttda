import math

import numpy as np
import tensorly as tl
import tensorly.decomposition
import tensorly.tenalg
from sklearn.base import BaseEstimator, TransformerMixin
from tensorly import random as tl_random

from hoda.tenalg import force_toeplitz, lobpcg, pinvh, trunc_gevd, trunc_svd


def solve_ratio_svd(scatter_w, scatter_b, r):
    v, _ = trunc_svd(pinvh(scatter_w) @ scatter_b, r)
    return v


def solve_ratio_gevd(scatter_w, scatter_b, r):
    v, _ = trunc_gevd(scatter_b, scatter_w, r)
    return v


def solve_ratio_lanczos(scatter_w, scatter_b, r):
    raise NotImplementedError


def solve_ratio_lobpcg(scatter_w, scatter_b, r, init=None, **solver_params):
    if init is None:
        init = tl.eye(scatter_w.shape[0])[:, :r]
    w, v = lobpcg(scatter_b, init, B=scatter_w, largest=True, **solver_params)
    return v


def solve_diff_svd(scatter_w, scatter_b, r, psi=1):
    phi = tl.trace(scatter_b) / tl.trace(scatter_w)
    v, w = trunc_svd(scatter_b - psi * phi * scatter_w, r)
    return v


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
        initialize="svd",
        shrinkage="oas",
        toeplitz=None,
        solver="ratio_gevd",
        verbose=False,
        solver_params=None,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.rank = rank
        self.initialize = initialize
        self.shrinkage = shrinkage
        self.toeplitz = toeplitz
        self.solver = solver
        self.verbose = verbose
        self.solver_params = solver_params

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

        # Initialize rank
        self.rank_ = self.rank
        if self.rank_ is None:
            self.rank_ = shape

        # Initialize projections
        self.projs_ = [None] * order
        for k in range(order):
            if self.initialize == "identity":
                self.projs_[k] = tl.eye(shape[k], self.rank_[k])
            elif self.initialize == "ones":
                self.projs_[k] = tl.ones((shape[k], self.rank_[k]))
            elif self.initialize == "random":
                self.projs_[k] = tl_random.random_tensor(
                    shape=(shape[k], self.rank_[k]),
                )
                self.projs_[k], _ = tl.qr(self.projs_[k], mode="reduced")
            elif self.initialize == "svd":
                x = tl.unfold(X, k + 1)
                self.projs_[k], _ = trunc_svd(x, self.rank_[k])
            else:
                raise ValueError(
                    "initialize should be one of {identity, ones, random, svd}"
                )

        # Calculate means and center
        class_means = []
        X_centered = []
        class_mean = 0
        for ci, c in enumerate(self.classes_):
            where = y == c
            where = where.reshape((where.shape[0], 1, 1))
            # mean = tl.mean(X_proj, axis=0, where=where)
            X_where = X[y == c]
            mean = tl.mean(X_where, axis=0)
            class_means += [mean]
            class_mean += mean / n_classes
            X_centered.append(X_where - mean)
        X_centered = tl.concatenate(X_centered, axis=0)
        for ci, c in enumerate(self.classes_):
            class_means[ci] -= class_mean

        # Find projections
        self.updates_ = []
        self.objective_ = []
        self.scatter_w_ = [None] * order
        self.scatter_b_ = [None] * order
        for self.iter_ in range(self.max_iter):
            if self.verbose:
                print(f"[{self.iter_}/{self.max_iter}]", end="  ")
            new_projs = [None] * order
            update = [0] * order
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
                scatter_w, shrinkage = self._shrink(
                    scatter_w, n_samples, self.shrinkage[k]
                )
                if self.verbose:
                    print(f"shrinkage={shrinkage:.4f}", end="  ")
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
                v = SOLVERS[self.solver](
                    scatter_w, scatter_b, self.rank_[k], **solver_params
                )
                # Orthonormalize for stability
                v, _ = tl.qr(v, mode="reduced")
                new_projs[k] = v

            # Stopping criterion
            break_flag = True
            for k in range(order):
                tol = self.tol * math.prod(self.projs_[k].shape)
                update[k] = tl.norm(new_projs[k] - self.projs_[k])
                if not update[k] < tol:
                    break_flag = False

            self.updates_.append(update)
            self.projs_ = new_projs
            if self.verbose:
                print(f"step={(sum(update)/order):.4e}")
            if break_flag:
                break
        self.updates_ = tl.tensor(self.updates_)
        return self

    def _shrink(self, scatter, n_epochs, shrinkage):
        n_features, _ = scatter.shape
        # Shrinkage regularization
        if shrinkage == "oas":
            shrinkage = oas(scatter, n_epochs)
        mu = tl.sum(tl.diag(scatter)) / n_features
        scatter = (1 - shrinkage) * scatter + shrinkage * mu * tl.eye(n_features)
        return scatter, shrinkage

    def transform(self, X, y=None):
        X = tl.tensor(X)
        order = len(X.shape) - 1
        X_trans = tl.tenalg.multi_mode_dot(
            X, self.projs_, modes=range(1, order + 1), transpose=True
        )
        return tl.to_numpy(X_trans)

    def inv_transform(self, X, y=None):
        X = tl.tensor(X)
        order = len(X.shape) - 1
        X_trans = tl.tenalg.multi_mode_dot(
            X, self.projs_, modes=range(1, order + 1), transpose=False
        )
        return X_trans


class BTTDA(BaseEstimator, TransformerMixin):
    def __init__(self, block_rank=None, hoda_params=None):
        self.hoda_params = hoda_params
        self.block_rank = block_rank

    def fit(self, X, y):
        X = X.copy()
        hoda_params = self.hoda_params
        if hoda_params is None:
            hoda_params = dict()

        self.blocks_ = []
        for b in range(self.block_rank):
            block = HODA(**hoda_params)
            block.fit(X, y)
            self.blocks_.append(block)
            X_approx = block.inv_transform(block.transform(X))
            X -= X_approx

    def transform(self, X, y=None):
        Xt = []
        for block in self.blocks_:
            Xt.append(block.transform(X, y))
        Xt = np.stack(Xt, axis=1)
        return Xt


def oas(emp_cov, n_samples):
    n_features = emp_cov.shape[0]
    mu = np.trace(emp_cov) / n_features

    # formula from Chen et al.'s **implementation**
    alpha = np.mean(emp_cov**2)
    num = alpha + mu**2
    den = (n_samples + 1.0) * (alpha - (mu**2) / n_features)

    shrinkage = np.real(num / den)
    return max(min(shrinkage, 1), 0)
