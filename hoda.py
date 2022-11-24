import math

import ipdb
import numpy as np
import scipy.linalg
import tensorly as tl
import tensorly.decomposition
import tensorly.tenalg
from sklearn.base import BaseEstimator, TransformerMixin
from tensorly import random as tl_random


class MLSVD(BaseEstimator, TransformerMixin):
    def __init__(self, rank=None):
        self.rank = rank

    def fit(self, X, y=None):
        modes = tuple(range(1, len(X.shape)))
        _, self.factors_ = tensorly.decomposition.partial_tucker(
            X, modes=modes, rank=self.rank
        )
        return self

    def transform(self, X, y=None):
        modes = tuple(range(1, len(X.shape)))
        Xt = tensorly.tenalg.multi_mode_dot(
            X, self.factors_, modes=modes, transpose=True
        )
        return Xt


class HODA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_iter=100,
        tol=1e-13,
        rank=None,
        initialize="identity",
        shrinkage="oas",
        toeplitz=None,
        solver="gevd",
        verbose=False,
        tl_context=None,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.rank = rank
        self.initialize = initialize
        self.shrinkage = shrinkage
        self.toeplitz = toeplitz
        self.solver = solver
        self.verbose = verbose
        self.tl_context = tl_context

    def fit(self, X, y):
        tl_context = self.tl_context
        if self.tl_context is None:
            tl_context = dict()
        X = tl.tensor(X, **tl_context)

        self.classes_, class_counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        shape = X.shape[1:]
        order = len(shape)

        # Initialize projections
        self.projs_ = [None] * order
        for k in range(order):
            if self.initialize == "identity":
                self.projs_[k] = tl.eye(shape[k], self.rank, **tl_context)
            elif self.initialize == "random":
                self.projs_[k] = tl_random.random_tensor(
                    shape=(shape[k], self.rank), **tl_context
                )
                self.projs_[k], _ = tl.qr(self.projs_[k], mode="reduced")
            elif self.initialize == "svd":
                x = tl.unfold(X, k + 1)
                self.projs_[k], _, _ = tl.partial_svd(x, n_eigenvecs=self.rank)
            else:
                raise ValueError("initialize should be one of {identity, random, svd}")

        # Find projections
        self.updates_ = []
        self.objective_ = []
        self.scatter_w_ = [None] * order
        self.scatter_b_ = [None] * order

        for self.iter_ in range(self.max_iter):
            if self.verbose:
                print(f"[{self.iter_}/{self.max_iter}]", end="  ")
            new_projs = [None] * order
            for k in range(order):
                modes = range(1, order + 1)
                X_proj = tl.tenalg.multi_mode_dot(
                    X, self.projs_, modes=modes, skip=k, transpose=True
                )
                # TODO: reshape is not necessary
                X_proj = tl.base.partial_unfold(X_proj, mode=k, skip_begin=1)

                class_means_proj = []
                X_proj_centered = []
                for ci, c in enumerate(self.classes_):
                    where = y == c
                    where = where.reshape((where.shape[0], 1, 1))
                    mean = tl.mean(X_proj, axis=0, where=where)
                    class_means_proj += [mean]
                    X_proj_where = X_proj[y == c]
                    X_proj_centered += [X_proj_where - mean]
                class_means_proj = tl.stack(class_means_proj, axis=0)

                # Calculate whithin class scatter
                scatter_w = 0
                for ci, c in enumerate(self.classes_):
                    scatter_w += tl.tensordot(
                        X_proj_centered[ci],
                        X_proj_centered[ci].conj(),
                        axes=([0, 2], [0, 2]),
                    )
                # Force symmetry
                scatter_w = (scatter_w + scatter_w.conj().T) / 2
                # Force toeplitz
                if self.toeplitz is not None and k in self.toeplitz:
                    scatter_w = self._make_toeplitz(scatter_w)
                # Shrink
                shrinkage = self.shrinkage
                if isinstance(shrinkage, tuple):
                    shrinkage = shrinkage[k]
                scatter_w = self._shrink(X_proj, scatter_w, shrinkage)
                self.scatter_w_[k] = scatter_w

                # Calculate between class scatter
                class_means_proj -= tl.mean(class_means_proj, axis=0)
                scatter_b = tl.zeros((shape[k], shape[k]))
                for c in range(n_classes):
                    scatter_b += (
                        class_means_proj[c]
                        @ class_means_proj[c].conj().T
                        * class_counts[c]
                    )
                # Force symmetry
                scatter_b = (scatter_b + scatter_b.conj().T) / 2
                self.scatter_b_[k] = scatter_b

                # Solve
                if self.solver == "gevd":
                    w, v = scipy.linalg.eigh(scatter_w, subset_by_value=[0, np.inf])
                    scatter_w = v @ tl.diag(w) @ v.conj().T
                    subset = [shape[k] - self.rank, shape[k] - 1]
                    w, v = scipy.linalg.eigh(
                        scatter_b, scatter_w, subset_by_index=subset
                    )
                elif self.solver == "sr":
                    raise NotImplementedError

                # Orthonormalize for stability
                v, _ = tl.qr(v, mode="reduced")
                new_projs[k] = v

            # Stopping criterion
            break_flag = True
            update = [0] * order
            for k in range(order):
                tol = self.tol * math.prod(self.projs_[k].shape)
                update[k] = tl.norm(new_projs[k] - self.projs_[k])
                self.updates_.append(update)
                if not update[k] < tol:
                    break_flag = False
            self.projs_ = new_projs
            if self.verbose:
                print(f"step={(sum(update)/order):.4e}")
            if break_flag:
                break
        self.updates_ = tl.tensor(self.updates_, **tl_context)
        return self

    def _make_toeplitz(self, scatter):
        n_features, _ = scatter.shape
        toep = [0] * n_features
        for f in range(n_features):
            toep[f] = tl.mean(tl.diag(scatter, k=f))
        cov_toep = scipy.linalg.toeplitz(toep)
        return cov_toep

    def _shrink(self, X_proj, scatter, shrinkage):
        n_features, _ = scatter.shape
        # Shrinkage regularization
        if self.shrinkage == "oas":
            shrinkage = oas(scatter, X_proj.shape[0])
        if self.verbose:
            print(f"shrinkage={shrinkage:.4f}", end="  ")
        mu = tl.sum(tl.diag(scatter)) / n_features
        scatter = (1 - shrinkage) * scatter + shrinkage * mu * tl.eye(n_features)
        return scatter

    def transform(self, X, y=None):
        order = len(X.shape) - 1
        X_trans = tl.tenalg.multi_mode_dot(
            X, self.projs_, modes=range(1, order + 1), transpose=True
        )
        X_trans = X_trans.reshape(X.shape[0], -1)
        return X_trans



def oas(emp_cov, n_samples):
    n_features = emp_cov.shape[0]
    mu = np.trace(emp_cov) / n_features

    # formula from Chen et al.'s **implementation**
    alpha = np.mean(emp_cov**2)
    num = alpha + mu**2
    den = (n_samples + 1.0) * (alpha - (mu**2) / n_features)

    shrinkage = np.real(num / den)
    return max(min(shrinkage, 1), 0)
