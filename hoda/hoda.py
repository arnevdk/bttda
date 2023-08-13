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

from hoda.tenalg import det, force_toeplitz, pinvh, trunc_eigh


def obj_ratio(scatter_b, scatter_t, _):
    """Ratio trace objective.

    Phan, A. H., & Cichocki, A. (2010).
    Tensor decompositions for feature extraction and classification of high
    dimensional datasets. Nonlinear theory and its applications, IEICE, 1(1), 37-68.
    """
    return scatter_b, scatter_t, True


def obj_diff(scatter_b, scatter_t, v, psi=1):
    """Trace ratio or difference objective.

    Phan, A. H., & Cichocki, A. (2010).
    Tensor decompositions for feature extraction and classification of high
    dimensional datasets. Nonlinear theory and its applications, IEICE, 1(1), 37-68.

    Wang, H., Yan, S., Xu, D., Tang, X., & Huang, T. (2007, June). Trace ratio
    vs. ratio trace for dimensionality reduction. In 2007 IEEE Conference on
    Computer Vision and Pattern Recognition (pp. 1-8). IEEE.
    """
    phi = tl.trace(v.T @ scatter_b @ v) / tl.trace(v.T @ scatter_t @ v)
    return scatter_b - psi * phi * scatter_t, None, True


def obj_lfl(scatter_b, scatter_t, v, psi=1):
    """Linear feature learning obbjective.

    Aghili, S. N., Kilani, S., Khushaba, R. N., & Rouhani, E. (2023).
    A spatial-temporal linear feature learning algorithm for P300-based -
    brain-computer interfaces. Heliyon, 9(4).
    """
    phi = tl.trace(v.T @ scatter_b @ v) / tl.trace(v.T @ scatter_t @ v)
    return scatter_b - psi * phi * scatter_t, scatter_t, True


def obj_od(scatter_b, scatter_t, v):
    """Optimal dimensionality discriminant analysis

    Nie, F., Xiang, S., Song, Y., & Zhang, C. (2007, April).
    Extracting the optimal dimensionality for discriminant analysis. In 2007
    IEEE International Conference on Acoustics, Speech and Signal Processing-ICASSP'07 (Vol. 2, pp. II-617). IEEE.

    Wang, J., Wang, L., Nie, F., & Li, X. (2021). A novel formulation of trace ratio linear discriminant analysis. IEEE Transactions on Neural Networks and Learning Systems, 33(10), 5568-5578.
    """
    s = tl.trace(v.T @ scatter_b @ v) / tl.trace(v.T @ scatter_t @ v)
    return s**2 * scatter_t - 2 * s * scatter_b, None, False


def obj_sr(
    scatter_w,
    scatter_t,
    v,
):
    """
    Idaji, M. J., Shamsollahi, M. B., & Sardouie, S. H. (2017).
    Higher order spectral regression discriminant analysis (HOSRDA): A tensor
    feature reduction method for ERP detection. Pattern Recognition, 70, 152-162.
    """
    raise NotImplementedError


OBJECTIVES = dict(
    ratio=obj_ratio,
    diff=obj_diff,
    lfl=obj_lfl,
    od=obj_od,
    sr=obj_sr,
)


def fisher_score(X, y):
    X = tl.tensor(X, dtype=X.dtype)
    n_samples, *shape = X.shape
    classes, class_counts = np.unique(y, return_counts=True)

    scatter_t = tl.norm(X, order=2) ** 2
    # Calculate class means, overall class mean and center data
    means, X_centered = center(X, y)
    class_mean = tl.mean(means, axis=0)

    # Calculate between class scatter
    scatter_b = 0
    for ci, c in enumerate(classes):
        scatter_b += class_counts[ci] * tl.norm(means[ci] - class_mean, order=2) ** 2
    # Calculate f score
    return float(scatter_b / scatter_t)


def center(X, y):
    _, *shape = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    means = tl.zeros((n_classes, *shape), dtype=X.dtype)
    X_centered = []
    for ci, c in enumerate(classes):
        where = y == c
        where = where.reshape((where.shape[0], 1, 1))
        X_where = X[y == c]
        means[ci] = tl.mean(X_where, axis=0)
        X_centered.append(X_where - means[ci])
    X_centered = tl.concatenate(X_centered, axis=0)
    return means, X_centered


def mode_scatter(X, k, weights=None, shrinkage=0, assume_centered=False):
    """Calculate the scatter matrix along a given tensor mode"""
    if weights is not None:
        X = (X.T * weights).T
    if shrinkage == "lw":
        X = tl.unfold(X, k + 1)
        scatter, shrinkage = ledoit_wolf(
            tl.to_numpy(X.T), assume_centered=assume_centered
        )

        scatter = tl.tensor(scatter)
        scatter *= X.shape[1] - 1
    elif shrinkage == "oas":
        X = tl.unfold(X, k + 1)
        scatter, shrinkage = oas(tl.to_numpy(X.T), assume_centered=assume_centered)
        scatter = tl.tensor(scatter)
        scatter *= X.shape[1] - 1
    else:
        order = len(X.shape[1:])
        modes = [0] + [kk + 1 for kk in range(order) if kk != k]
        if assume_centered:
            X = X - tl.mean(X, axis=0)
        scatter = tl.tensordot(X, X.conj(), axes=(modes, modes))
        scatter = (1 - shrinkage) * scatter + shrinkage * tl.mean(
            tl.diag(scatter)
        ) * tl.eye(scatter.shape[0], dtype=X.dtype)
    return scatter, shrinkage


def mse(A, B):
    return float(np.abs(tl.mean((B - A) ** 2)))


class HODA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(
        self,
        max_iter=100,
        tol=1e-12,
        rank=None,
        init="mlsvd",
        shrinkage="oas",
        toeplitz=None,
        taper=False,
        obj="ratio",
        solver="lanczos",
        verbose=False,
        solver_params=None,
        keep_train_info=False,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.rank = rank
        self.init = init
        self.shrinkage = shrinkage
        self.toeplitz = toeplitz
        self.obj = obj
        self.solver = solver
        self.verbose = verbose
        self.solver_params = solver_params
        self.keep_train_info = keep_train_info
        self.taper = taper

    def fit(self, X, y):
        # Setup
        X = tl.tensor(X.copy(), dtype=X.dtype)
        self.classes_, class_counts = np.unique(y, return_counts=True)
        class_counts = tl.tensor(class_counts)
        n_samples, *shape = X.shape
        order = len(shape)
        if self.rank is None:
            self.rank_ = shape.copy()
        else:
            self.rank_ = self.rank

        # Initialize solver.T
        if self.obj not in OBJECTIVES.keys():
            raise ValueError(f"objective must be one of {list(OBJECTIVES.keys())}")
        solver_params = self.solver_params
        if solver_params is None:
            solver_params = dict()

        # Calculate means and center
        means, X_centered = center(X, y)
        means_centered = means - np.mean(means, axis=0)

        # Initialize rank
        if self.rank is None:
            for k in range(order):
                # Calculate within class scatter
                scatter_w, _ = mode_scatter(
                    X_centered, k, assume_centered=True, shrinkage=self.shrinkage[k]
                )
                if self.toeplitz is not None and k in self.toeplitz:
                    scatter_w = force_toeplitz(scatter_w, taper=self.taper)
                # Calculate between class scatter
                scatter_b, _ = mode_scatter(
                    means_centered, k, weights=tl.sqrt(class_counts)
                )
                # Solve
                last_obj = np.inf

                for r in range(1, shape[k] + 1):
                    u = tl.eye(shape[k], dtype=X.dtype)[:, :r]
                    last_u = u
                    for t in range(self.max_iter):
                        A, B, largest = obj_od(scatter_b, scatter_w, u)
                        u, w = trunc_eigh(
                            A,
                            B,
                            init=u,
                            r=r,
                            largest=largest,
                            method=self.solver,
                            **solver_params,
                        )

                        if tl.norm(u - last_u, order=2) < self.tol:
                            break
                        last_u = u
                    s = tl.trace(u.T @ scatter_b @ u)
                    s /= tl.trace(u.T @ scatter_w @ u)
                    obj = s**2 * tl.trace(u.T @ scatter_w @ u) - 2 * s * tl.trace(
                        u.T @ scatter_b @ u
                    )
                    if obj > last_obj:
                        self.rank_[k] = r - 1
                        break
                    last_obj = obj

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

        # Initialize scatter matrices
        self.scatter_w_ = [None] * order
        self.scatter_b_ = [None] * order
        self.cov_w_ = [None] * order
        self.cov_l_ = []
        self.cov_l_inv_ = []
        self.scatter_t_ = []
        for k in range(order):
            self.cov_l_.append(tl.eye(self.rank_[k]))
            self.cov_l_inv_.append(tl.eye(self.rank_[k]))
            self.scatter_t_.append(mode_scatter(X_centered, k, assume_centered=True)[0])

        # Initialze training information
        if self.keep_train_info:
            self.train_info_ = dict(
                mode_objective=[[] for _ in range(order)],
                mode_update=[[] for _ in range(order)],
                mode_shrinkage=[[] for _ in range(order)],
                f_score=[],
                mse=[],
            )

        # Iteratively find projections
        if self.verbose:
            print(f"Fitting discriminative tucker model of rank {self.rank_} ...")

        iterator = range(self.max_iter)
        if self.verbose:
            iterator = tqdm(iterator)
        for self.iter_ in iterator:
            update = 0
            for k in range(order):
                modes = range(1, order + 1)
                X_centered_proj = tl.tenalg.multi_mode_dot(
                    X_centered,
                    self.scalings_,
                    modes=modes,
                    skip=k,
                    transpose=True,
                )
                scatter_w, shrinkage = mode_scatter(
                    X_centered_proj,
                    k,
                    assume_centered=True,
                    shrinkage=self.shrinkage[k],
                )
                if self.toeplitz is not None and k in self.toeplitz:
                    scatter_w = force_toeplitz(scatter_w, taper=self.taper)
                self.scatter_w_[k] = scatter_w

                # Calculate between class scatter
                means_centered_proj = tl.tenalg.multi_mode_dot(
                    means_centered, self.scalings_, modes=modes, skip=k, transpose=True
                )
                scatter_b, _ = mode_scatter(
                    means_centered_proj, k, weights=tl.sqrt(class_counts)
                )
                self.scatter_b_[k] = scatter_b

                # Solve
                A, B, largest = OBJECTIVES[self.obj](
                    scatter_b, scatter_w, self.scalings_[k]
                )
                u, w = trunc_eigh(
                    A,
                    B,
                    init=self.scalings_[k],
                    r=self.rank_[k],
                    method=self.solver,
                    largest=largest,
                    **solver_params,
                )
                # Why this line?
                # u, w, _ = tensorly.tenalg.svd_interface(
                #    u @ u.T @ self.scatter_t_[k] @ u @ u.T,
                #    n_eigenvecs=self.rank_[k],
                #    flip_sign=True,
                # )
                u, _ = tl.qr(u, mode="reduced")

                obj = tl.sum(w)

                mode_update = tl.norm(u - self.scalings_[k], order=2)
                update += mode_update
                # Store mode training information
                if self.keep_train_info:
                    self.train_info_["mode_objective"][k].append(float(obj))
                    self.train_info_["mode_update"][k].append(float(mode_update))
                    self.train_info_["mode_shrinkage"][k].append(float(shrinkage))

                self.scalings_[k] = u
                self.weightings_[k] = w
            # Calculate latent factor covariance
            core = self.transform(X, y)
            core_centered = core.copy()
            for c in self.classes_:
                core_centered[y == c] -= tl.mean(core[y == c], axis=0)
            for k in range(order):
                self.cov_w_[k] = self.scatter_w_[k] / (
                    tl.prod(core.shape) / core.shape[k + 1] - 1
                )

                scatter_l, _ = mode_scatter(core_centered, k, assume_centered=True)
                cov_l = scatter_l / (tl.prod(core.shape) / core.shape[k + 1] - 1)
                self.cov_l_[k] = cov_l
                self.cov_l_inv_[k] = pinvh(cov_l)

            # Store iteration training information
            if self.keep_train_info:
                f_score = fisher_score(core, y)
                X_rec = self.inv_transform(core)
                self.train_info_["f_score"].append(f_score)
                self.train_info_["mse"].append(mse(X, X_rec))

            # Check convergence
            if update < self.tol:
                break

        return self

    def transform(self, X, y=None):
        X = tl.tensor(X.copy(), dtype=X.dtype)
        order = len(X.shape) - 1
        Xt = tl.tenalg.multi_mode_dot(
            X, self.scalings_, modes=range(1, order + 1), transpose=True
        )
        return Xt

    def inv_transform(self, Xt, y=None):
        Xt = tl.tensor(Xt.copy(), dtype=Xt.dtype)
        order = len(Xt.shape) - 1
        activation_patterns = []
        for k in range(order):
            # Haufe method
            ap = self.cov_w_[k] @ self.scalings_[k] @ self.cov_l_inv_[k]
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
        self.classes_, class_counts = np.unique(y, return_counts=True)

        hoda_params = self.hoda_params
        if hoda_params is None:
            hoda_params = dict()

        if self.keep_train_info:
            self.train_info_ = dict(
                f_score=[],
                mse=[],
            )

        self.blocks_ = []
        n_blocks = self.n_blocks
        X_rec = np.zeros_like(X)
        X_defl = X.copy()
        # Deflation scheme
        for b in range(n_blocks):
            if self.verbose:
                print(f"Fitting block {b+1}/{self.n_blocks}...")
            # Fit Tucker block
            block = HODA(**hoda_params)
            block.fit(X_defl, y)
            self.blocks_.append(block)
            # Reconstruct and subtract for next iteration
            X_approx = block.inv_transform(block.transform(X_defl))
            X_defl -= X_approx
            X_rec += X_approx
            if self.verbose:
                print()
            if self.keep_train_info:
                Xt = self.transform(X)
                self.train_info_["f_score"].append(fisher_score(Xt, y))
                self.train_info_["mse"].append(mse(X, X_rec))

        return self

    def transform(self, X, y=None, n_blocks=None):
        X = tl.tensor(X.copy(), dtype=X.dtype)
        n_samples, *_ = X.shape
        Xt = []
        if n_blocks is None:
            n_blocks = len(self.blocks_)
        for b in range(n_blocks):
            block = self.blocks_[b]
            Xtb = block.transform(X, y)
            Xt.append(Xtb.reshape(n_samples, -1))
        Xt = tl.concatenate(Xt, axis=1)
        return Xt

    def inv_transform():
        pass
