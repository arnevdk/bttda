import cupy
import ipdb
import numpy as np
import scipy.linalg
import tensorly as tl
import tensorly.decomposition
import tensorly.tenalg
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.covariance import ledoit_wolf, oas, shrunk_covariance
from sklearn.linear_model import ElasticNet
from tensorly import random as tl_random
from tqdm.notebook import tqdm

from hoda.tenalg import det, force_toeplitz, lobpcg, pinvh, trunc_eigh


def solve_ratio_svd(scatter_b, scatter_t, v, r):
    v, w, _ = tensorly.tenalg.svd_interface(pinvh(scatter_t) @ scatter_b, n_eigenvecs=r)
    o = tl.trace(v.T @ scatter_b @ v) / tl.trace(v.T @ scatter_t @ v)
    return v, w, np.real(o)


def solve_ratio_gevd(scatter_b, scatter_t, v, r):
    v, w = trunc_eigh(scatter_b, scatter_t, r)
    return v, w


def solve_ratio_lanczos(scatter_b, scatter_t, v, r):
    raise NotImplementedError


def solve_ratio_lobpcg(scatter_b, scatter_t, v, r, **solver_params):
    solver_params["max_iter"] = 1
    w, v = lobpcg(scatter_b, v, B=scatter_t, largest=True, **solver_params)
    return v, w


def solve_diff(scatter_b, scatter_t, v, r, psi=1):
    phi = tl.trace(v.T @ scatter_b @ v) / tl.trace(v.T @ scatter_t @ v)
    v, w = trunc_eigh(scatter_b - psi * phi * scatter_t, r=r, largest=True)
    o = tl.sum(w)
    return v, w, o


def solve_lfl(scatter_b, scatter_t, v, r, psi=1):
    phi = tl.trace(v.T @ scatter_b @ v) / tl.trace(v.T @ scatter_t @ v)
    v, w = trunc_eigh(scatter_b - psi * phi * scatter_t, B=scatter_t, r=r, largest=True)
    o = tl.sum(w)
    return v, w, o


def solve_od(scatter_b, scatter_t, v, r):

    s = tl.trace(v.T @ scatter_b @ v) / tl.trace(v.T @ scatter_t @ v)
    w, v = cupy.linalg.eigh(-(s**2 * scatter_t - 2 * s * scatter_b))
    idc = np.argsort(-w)[:r]
    w = w[idc]
    v = v[:, idc]
    o = tl.sum(w)
    return v, w, o


def solve_sr(scatter_w, scatter_t, v, w, r):
    raise NotImplementedError


SOLVERS = dict(
    ratio_svd=solve_ratio_svd,
    ratio_gevd=solve_ratio_gevd,
    ratio_lanczos=solve_ratio_lanczos,
    ratio_lobpcg=solve_ratio_lobpcg,
    diff=solve_diff,
    sr=solve_sr,
    od=solve_od,
    lfl=solve_lfl,
)


def fisher_score(X, y):
    X = tl.tensor(X, dtype=X.dtype)
    n_samples, *shape = X.shape
    classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(classes)

    # Calculate class means, overall class mean and center data
    means = tl.zeros((n_classes, *shape), dtype=X.dtype)
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


class HODA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(
        self,
        max_iter=100,
        tol=1e-12,
        rank=None,
        init="mlsvd",
        shrinkage="oas",
        toeplitz=None,
        solver="ratio_gevd",
        verbose=False,
        solver_params=None,
        keep_train_info=False,
        priors=None,
        taper=False,
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
        self.keep_train_info = keep_train_info
        self.priors = priors
        self.taper = taper

    def fit(self, X, y):
        X = tl.tensor(X, dtype=X.dtype)
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
        # scatter_t_orig = [None] * order
        # for k in range(order):
        #    modes = [0] + [kk + 1 for kk in range(order) if kk != k]
        #    scatter_t_orig[k] = tl.tensordot(X, X, axes=(modes, modes))

        # Calculate means and center
        means = tl.zeros((n_classes, *shape), dtype=X.dtype)
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
            # TODO: rename to means_centered

        if self.rank is None:
            self.rank_ = shape.copy()
        else:
            self.rank_ = self.rank

        # Initialize projections
        if self.verbose:
            print("Initializing factors...")
        self.scalings_ = [None] * order
        self.weightings_ = [None] * order
        if self.init == "mlsvd":
            modes = tuple(range(1, order + 1))
            (_, self.scalings_), _ = tl.decomposition.partial_tucker(
                X_centered,
                rank=self.rank_,
                modes=modes,
            )
        else:
            for k in range(order):
                if self.init == "identity":
                    self.scalings_[k] = tl.eye(shape[k], self.rank_[k], dtype=X.dtype)
                elif self.init == "ones":
                    self.scalings_[k] = tl.ones(
                        (shape[k], self.rank_[k]), dtype=X.dtype
                    )
                elif self.init == "random":
                    self.scalings_[k] = tl_random.random_tensor(
                        shape=(shape[k], self.rank_[k]),
                    )
                    self.scalings_[k], _ = tl.qr(self.scalings_[k], mode="reduced")
                elif self.init == "eye":
                    self.scalings_[k] = tl.eye(shape[k], dtype=X.dtype)[
                        :, : self.rank_[k]
                    ]
                elif self.init == "svd":
                    Xk = tl.unfold(X, k + 1)
                    self.scalings_[k], _, _ = tensorly.tenalg.svd_interface(
                        Xk, n_eigenvecs=self.rank_[k]
                    )
                else:
                    raise ValueError(
                        "init should be one of {identity, ones, random, svd}"
                    )

        # Initialize rank
        if self.rank is None:
            for k in range(order):
                modes = range(1, order + 1)
                X_proj = tl.tenalg.multi_mode_dot(
                    X_centered,
                    self.scalings_,
                    modes=modes,
                    skip=k,
                    transpose=True,
                )
                scatter_w = self._scatter_w(X_proj, k)

                # Calculate between class scatter
                means_proj = tl.tenalg.multi_mode_dot(
                    means, self.scalings_, modes=modes, skip=k, transpose=True
                )
                scatter_b = self._scatter_b(means_proj, class_counts, k)

                # Solve
                scatter_t = scatter_w + scatter_b
                u, w, obj = SOLVERS[self.solver](
                    scatter_b,
                    scatter_t,
                    self.scalings_[k],
                    self.rank_[k],
                    **solver_params,
                )
                u = u[:, w > 0]
                self.rank_[k] = u.shape[1]
                self.scalings_[k] = self.scalings_[k][:, : self.rank_[k]]

        if self.verbose:
            print(f"Fitting discriminative tucker model of rank {self.rank_} ...")

        # Store initial training information
        if self.keep_train_info:
            core = self.transform(X, y)
            f_score = fisher_score(core, y)
            # mse = np.real(tl.mean((self.inv_transform(core) - X) ** 2))
            mse = 0
            # TODO: mode f score
            self.train_info_ = dict(
                mode_objective=[[] for _ in range(order)],
                mode_update=[[] for _ in range(order)],
                f_score=[float(f_score)],
                mse=[],
            )

        # Iteratively find projections
        self.scatter_w_ = [None] * order
        self.scatter_b_ = [None] * order
        iterator = range(self.max_iter)
        if self.verbose:
            iterator = tqdm(iterator)
        for self.iter_ in iterator:
            update = 0
            for k in range(order):
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
                u, w, obj = SOLVERS[self.solver](
                    scatter_b,
                    scatter_w,
                    self.scalings_[k],
                    self.rank_[k],
                    **solver_params,
                )
                u, w, _ = tensorly.tenalg.svd_interface(
                    u @ u.T @ scatter_w @ u @ u.T,
                    n_eigenvecs=self.rank_[k],
                    flip_sign=True,
                )

                u, _ = tl.qr(u, mode="reduced")

                mode_update = tl.norm(u - self.scalings_[k], order=2) / (
                    shape[k] * self.rank_[k] * (tl.norm(X) / np.prod(X.shape))
                )
                update += mode_update
                # Store mode training information
                if self.keep_train_info:
                    self.train_info_["mode_objective"][k].append(float(obj))
                    self.train_info_["mode_update"][k].append(float(mode_update))

                # Elastic net
                # u = self.scalings_[k]
                # mu = tl.trace(u.T @ scatter_w @ u) / tl.trace(u.T @ scatter_b @ u)
                # Phi, l, _ = trunc_svd(scatter_w - mu * scatter_b, shape[k])
                # Xk = Phi @ tl.diag(tl.sign(l) * tl.sqrt(tl.abs(l))) @ Phi.T

                # A = tl.ones((shape[k], shape[k]))
                # en = ElasticNet(alpha=1.0, l1_ratio=1, fit_intercept=False)
                # for i_en in range(100):
                #    en.fit(cupy.asnumpy(A @ Xk.T), cupy.asnumpy(Phi.T @ Xk.T))
                #    u = tl.tensor(en.coef_)
                #    u, w, v = trunc_svd(Phi.T @ Xk @ Xk.T @ u, shape[k])
                #    u = u[:, self.rank_[k] :]
                #    v = v[:, self.rank_[k] :]
                #    w = w[self.rank_[k] :]
                #    A = u @ v.T

                # Sparse core

                # core = tl.tenalg.mode_dot(X_proj, u.T, k + 1)
                # samples = []
                # for c in self.classes_:
                #    samples.append(cupy.asnumpy(core[y == c]))
                # F, p = scipy.stats.f_oneway(*samples, axis=0)
                # core[:, p > 1] = 0
                # core_flat = cupy.asnumpy(tl.unfold(core, k + 1))
                # core_flat = core_flat[~np.all(core_flat == 0, axis=1), :]
                # X_proj_flat = cupy.asnumpy(tl.unfold(X_proj, k + 1))
                # u = scipy.linalg.lstsq(core_flat.T, X_proj_flat.T)[0].T
                # self.rank_[k] = u.shape[1]
                # u = tl.tensor(u)

                self.scalings_[k] = u
                self.weightings_[k] = w

            # Store iteration training information
            core = self.transform(X, y)
            if self.keep_train_info:
                f_score = fisher_score(core, y)
                mse = np.abs(tl.mean((self.inv_transform(core) - X) ** 2))
                self.train_info_["f_score"].append(float(f_score))
                self.train_info_["mse"].append(float(mse))

            # Check convergence
            if update < self.tol:
                break

            # Calculate latent factor scatter
            self.cov_l_ = []
            self.cov_l_inv_ = []
            # TODO: make cov function
            core_centered = core.copy()
            for c in self.classes_:
                core_centered[y == c] -= tl.mean(core[y == c], axis=0)
            for k in range(order):
                # TODO: tensor contraction instead of unfolding
                core_c_k = tl.unfold(core_centered, k + 1)
                cov = (core_c_k @ core_c_k.conj().T) / (core_c_k.shape[-1] - 1)
                self.cov_l_.append(cov)
                self.cov_l_inv_.append(pinvh(cov))

        return self

    def _scatter_w(self, X, k):
        # Calculate whithin class scatter with shrinkage regularization
        # TODO: tensor contraction instead of unfolding
        X = tl.unfold(X, k + 1)
        # if self.shrinkage is not None:
        #    if self.shrinkage[k] == "lw":
        #        gamma = None
        #    else:
        #        gamma = self.shrinkage[k]
        #    scatter_w, s = shrinkage(tl.to_numpy(X), standardize=True, gamma=gamma)
        # else:
        #    scatter_w = X @ X.conj().T

        if self.shrinkage[k] == "lw":
            scatter_w, _ = ledoit_wolf(tl.to_numpy(X.T), assume_centered=True)
        elif self.shrinkage[k] == "oas":
            scatter_w, _ = oas(tl.to_numpy(X.T), assume_centered=True)
        else:
            # order = len(X.shape[1:])
            # modes = [0] + [kk + 1 for kk in range(order) if kk != k]
            # scatter_w = tl.tensordot(X, X.conj(), axes=(modes, modes))
            scatter_w = X @ X.conj().T
            # scatter_w = shrunk_covariance(tl.to_numpy(scatter_w), self.shrinkage[k])

            scatter_w = (1 - self.shrinkage[k]) * scatter_w + self.shrinkage[
                k
            ] * tl.mean(tl.diag(scatter_w)) * tl.eye(scatter_w.shape[0], dtype=X.dtype)

        scatter_w = tl.tensor(scatter_w, dtype=X.dtype)
        # Force Toeplitz structure
        if self.toeplitz is not None and k in self.toeplitz:
            scatter_w = force_toeplitz(scatter_w, taper=self.taper)
        return scatter_w

    def _scatter_b(self, means, class_counts, k):
        order = len(means.shape[1:])
        n_classes, *shape = means.shape
        # Calculate between class scatter
        modes = [kk for kk in range(order) if kk != k]
        scatter_b = 0
        for ci, c in enumerate(self.classes_):

            scatter_b += (
                tl.tensordot(means[ci], means[ci].conj(), axes=(modes, modes))
                * class_counts[ci]
            )
        return scatter_b

    def transform(self, X, y=None):
        X = tl.tensor(X)
        order = len(X.shape) - 1
        Xt = tl.tenalg.multi_mode_dot(
            X, self.scalings_, modes=range(1, order + 1), transpose=True
        )
        return Xt

    def inv_transform(self, Xt, y=None):
        Xt = tl.tensor(Xt)
        order = len(Xt.shape) - 1
        activation_patterns = []
        for k in range(order):
            # Haufe method
            ap = self.scatter_w_[k] @ self.scalings_[k] @ self.cov_l_inv_[k]
            activation_patterns.append(ap)
        X = tl.tenalg.multi_mode_dot(
            Xt, activation_patterns, modes=range(1, order + 1), transpose=False
        )
        return X


class BTTDA(BaseEstimator, TransformerMixin):
    def __init__(
        self, n_blocks=None, hoda_params=None, verbose=False, keep_train_info=False
    ):
        self.hoda_params = hoda_params
        self.n_blocks = n_blocks
        self.verbose = verbose
        self.keep_train_info = keep_train_info

    def fit(self, X, y):
        X = tl.tensor(X.copy(), dtype=X.dtype)
        X_orig = X.copy()
        self.classes_, class_counts = np.unique(y, return_counts=True)

        hoda_params = self.hoda_params
        if hoda_params is None:
            hoda_params = dict()

        self.blocks_ = []
        self.train_info_ = dict(
            mse=[],
            f_score=[],
        )
        self.train_mse_ = []
        X_rec = 0
        last_f_score = 0
        n_blocks = self.n_blocks
        if n_blocks is None:
            n_blocks = 512
        for b in range(n_blocks):
            if self.verbose:
                print(f"Fitting block {b+1}/{self.n_blocks}...")
            # Fit Tucker block
            block = HODA(**hoda_params)
            block.fit(X, y)
            self.blocks_.append(block)
            # Reconstruct and subtract for next iteration
            X_approx = block.inv_transform(block.transform(X))
            X -= X_approx
            if self.verbose:
                print()
            X_rec += X_approx
            mse = np.real(tl.mean((X_orig - X_rec) ** 2))
            f_score = fisher_score(self.transform(X), y)
            if self.n_blocks is None and f_score <= last_f_score:
                self.blocks_.pop()
                break
            last_f_score = f_score
            if self.keep_train_info:
                self.train_info_["mse"].append(float(mse))
                self.train_info_["f_score"].append(f_score)

        return self

    def _transform(self, X, y=None):
        X = X.copy()
        n_samples, *_ = X.shape
        Xt = []
        for block in self.blocks_:
            Xtb = block.transform(X, y)
            Xrb = block.inv_transform(Xtb)
            # X -= Xrb
            Xt.append(Xtb.reshape(n_samples, -1))
        Xt = tl.concatenate(Xt, axis=1)
        return Xt

    def transform(self, X, y=None):
        X = tl.tensor(X, dtype=X.dtype)
        Xt = self._transform(X, y=None)
        Xt = tl.to_numpy(Xt)
        return Xt
