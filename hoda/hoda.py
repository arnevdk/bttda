import math

import cupyx.scipy.linalg
import cupyx.scipy.sparse.linalg
import ipdb
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import tensorly as tl
import tensorly.decomposition
import tensorly.tenalg
from sklearn.base import BaseEstimator, TransformerMixin
from tensorly import random as tl_random


class MLSVD(BaseEstimator, TransformerMixin):
    def __init__(self, modes=None, rank=None):
        self.modes = modes
        self.rank = rank

    def fit(self, X, y=None):
        shape = X.shape[1:]
        order = len(shape)
        modes = self.modes
        if modes is None:
            modes = np.arange(order)
        else:
            modes = np.asarray(modes)

        self.factors_ = [np.eye(shape[k]) for k in range(order)]
        _, factors = tensorly.decomposition.partial_tucker(
            X, modes=modes + 1, rank=self.rank
        )
        for mi, mode in enumerate(modes):
            self.factors_[mode] = factors[mi]
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
        initialize="ones",
        shrinkage="oas",
        toeplitz=None,
        solver="ratio-gevd",
        verbose=False,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.rank = rank
        self.initialize = initialize
        self.shrinkage = shrinkage
        self.toeplitz = toeplitz
        self.solver = solver
        self.verbose = verbose

    def fit(self, X, y):
        X = tl.tensor(X)
        self.classes_, class_counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        shape = X.shape[1:]
        order = len(shape)

        # Initialize projections
        self.projs_ = [None] * order
        for k in range(order):
            if self.initialize == "identity":
                self.projs_[k] = tl.eye(shape[k], self.rank[k])
            elif self.initialize == "ones":
                self.projs_[k] = tl.eye(shape[k], self.rank[k])
            elif self.initialize == "random":
                self.projs_[k] = tl_random.random_tensor(
                    shape=(shape[k], self.rank[k]),
                )
                self.projs_[k], _ = tl.qr(self.projs_[k], mode="reduced")
            elif self.initialize == "svd":
                x = tl.unfold(X, k + 1)
                self.projs_[k], _, _ = tl.partial_svd(x, n_eigenvecs=self.rank[k])
            else:
                raise ValueError("initialize should be one of {identity, random, svd}")

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

        for ci, c in enumerate(self.classes_):
            class_means[ci] -= class_mean

        # Find projections
        self.updates_ = []
        self.objective_ = []
        self.scatter_w_ = [None] * order
        self.scatter_b_ = [None] * order

        scatter_t = [None] * order
        for k in range(order):
            Xk = tl.base.unfold(X, mode=k + 1)
            scatter_t[k] = Xk @ Xk.conj().T

        for self.iter_ in range(self.max_iter):
            if self.verbose:
                print(f"[{self.iter_}/{self.max_iter}]", end="  ")
            new_projs = [None] * order
            update = [0] * order
            for k in range(order):
                # Calculate whithin class scatter
                scatter_w = 0
                modes = range(1, order + 1)
                for ci, c in enumerate(self.classes_):
                    X_proj_c = tl.tenalg.multi_mode_dot(
                        X_centered[ci], self.projs_, modes=modes, skip=k, transpose=True
                    )
                    X_proj_c = tl.base.unfold(X_proj_c, mode=k + 1)
                    scatter_w += X_proj_c @ X_proj_c.conj().T
                # Force symmetry
                scatter_w = (scatter_w + scatter_w.conj().T) / 2
                # Force toeplitz
                if self.toeplitz is not None and k in self.toeplitz:
                    scatter_w = self._make_toeplitz(scatter_w)
                # Shrink
                shrinkage = self.shrinkage
                if isinstance(shrinkage, tuple):
                    shrinkage = shrinkage[k]
                scatter_w, shrinkage = self._shrink(X.shape[0], scatter_w, shrinkage)
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
                """
                Multilinear Discriminant Analysis for
                Higher-Order Tensor Data Classification
                Qun Li, Member, IEEE and Dan Schonfeld, Fellow, IEE
                """
                if self.solver == "ratio-gevd":
                    # Optimize scatter difference criterion
                    w, v = scipy.linalg.eigh(scatter_w, subset_by_value=[0, np.inf])
                    scatter_w = v @ tl.diag(w) @ v.conj().T
                    subset = [shape[k] - self.rank[k], shape[k] - 1]
                    w, v = scipy.linalg.eigh(
                        scatter_b, scatter_w, subset_by_index=subset
                    )
                elif self.solver == "ratio-gevd-lanczos":
                    raise NotImplementedError
                elif self.solver == "ratio-gevd-lobpcg":
                    if tl.get_backend() == "cupy":
                        w, v = cupyx.scipy.sparse.linalg.lobpcg(
                            scatter_b,
                            self.projs_[k],
                            B=scatter_w,
                            largest=True,
                            maxiter=1,
                        )
                    elif tl.get_backend() == "numpy":
                        w, v = scipy.sparse.linalg.lobpcg(
                            scatter_b,
                            self.projs_[k],
                            B=scatter_w,
                            largest=True,
                            maxiter=1,
                        )
                    else:
                        raise NotImplementedError

                elif self.solver == "sr":
                    raise NotImplementedError
                elif self.solver == "diff-svd":
                    # Optimize scatter difference criterion
                    phi = tl.trace(scatter_b) / tl.trace(scatter_w)
                    v, w, _ = tl.partial_svd(
                        scatter_b - phi * scatter_w, n_eigenvecs=self.rank[k]
                    )
                    v, w, _ = tl.partial_svd(
                        v @ v.conj().T @ scatter_t[k] @ v @ v.conj().T,
                        n_eigenvecs=self.rank[k],
                    )
                    pass
                elif self.solver == "ratio-svd":
                    raise NotImplementedError
                else:
                    raise ValueError

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

    def _make_toeplitz(self, scatter):
        n_features, _ = scatter.shape
        toep = [0] * n_features
        for f in range(n_features):
            toep[f] = tl.mean(tl.diag(scatter, k=f))
        taper = tl.arange(len(toep), 0, -1) - 1
        toep = tl.tensor(toep) * taper
        toep = tl.tensor(toep)
        if tl.get_backend() == "numpy":
            cov_toep = scipy.linalg.toeplitz(toep)
        elif tl.get_backend() == "cupy":
            cov_toep = cupyx.scipy.linalg.toeplitz(toep)
        else:
            raise NotImplementedError
        return cov_toep

    def _shrink(self, n_epochs, scatter, shrinkage):
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


def oas(emp_cov, n_samples):
    n_features = emp_cov.shape[0]
    mu = np.trace(emp_cov) / n_features

    # formula from Chen et al.'s **implementation**
    alpha = np.mean(emp_cov**2)
    num = alpha + mu**2
    den = (n_samples + 1.0) * (alpha - (mu**2) / n_features)

    shrinkage = np.real(num / den)
    return max(min(shrinkage, 1), 0)
