import math

import ipdb
import numpy as np
import pandas as pd
import tensorly as tl
import tensorly.decomposition
import tensorly.tenalg
from sklearn.base import (BaseEstimator, ClassifierMixin, TransformerMixin,
                          clone)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFwe
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from tensorly import random as tl_random
from tqdm.notebook import tqdm

import hoda.backend as backend
from hoda.classification import Vectorize
from hoda.cov import KroneckerCovariance, center, ledoit_wolf_shrinkage
from hoda.gpu_opt import combine_pvalues

# from sklearn.feature_selection import f_classif


def obj_rt(scatter_b, scatter_w, _):
    """Ratio trace objective Tr(uT Sw^-1 Sb u)

    Phan, A. H., & Cichocki, A. (2010).
    Tensor decompositions for feature extraction and classification of high
    dimensional datasets. Nonlinear theory and its applications, IEICE, 1(1), 37-68.

    Wang, H., Yan, S., Xu, D., Tang, X., & Huang, T. (2007, June). Trace ratio
    vs. ratio trace for dimensionality reduction. In 2007 IEEE Conference on
    Computer Vision and Pattern Recognition (pp. 1-8). IEEE.

    """
    return scatter_b, scatter_w, True


def obj_tr(scatter_b, scatter_w, u, psi=1):
    """Trace ratio objective Tr(uT Sb u)/Tr(uT Sw u)

    Phan, A. H., & Cichocki, A. (2010).
    Tensor decompositions for feature extraction and classification of high
    dimensional datasets. Nonlinear theory and its applications, IEICE, 1(1), 37-68.

    Wang, H., Yan, S., Xu, D., Tang, X., & Huang, T. (2007, June). Trace ratio
    vs. ratio trace for dimensionality reduction. In 2007 IEEE Conference on
    Computer Vision and Pattern Recognition (pp. 1-8). IEEE.
    """
    phi = tl.trace(u.T @ scatter_b @ u) / tl.trace(u.T @ scatter_w @ u)
    # _, phi = trunc_eigh(
    #    backend.np.linalg.pinv(scatter_w) @ scatter_b,
    #    rank=1,
    #    method="lanczos",
    #    largest=True,
    # )
    A = scatter_b - psi * phi * scatter_w
    return A, None, True


def obj_lfl(scatter_b, scatter_w, u):
    """Linear feature learning obbjective.

    Aghili, S. N., Kilani, S., Khushaba, R. N., & Rouhani, E. (2023).
    A spatial-temporal linear feature learning algorithm for P300-based -
    brain-computer interfaces. Heliyon, 9(4).
    """
    raise NotImplementedError


def obj_od(scatter_b, scatter_w, u):
    """Optimal dimensionality discriminant analysis

    Nie, F., Xiang, S., Song, Y., & Zhang, C. (2007, April).
    Extracting the optimal dimensionality for discriminant analysis. In 2007
    IEEE International Conference on Acoustics, Speech and Signal Processing-ICASSP'07 (Vol. 2, pp. II-617). IEEE.

    Wang, J., Wang, L., Nie, F., & Li, X. (2021). A novel formulation of trace ratio linear discriminant analysis. IEEE Transactions on Neural Networks and Learning Systems, 33(10), 5568-5578.
    """
    s = tl.trace(u.T @ scatter_b @ u) / tl.trace(u.T @ scatter_w @ u)
    return s**2 * scatter_w - 2 * s * scatter_b, None, False


def obj_sr(
    scatter_b,
    scatter_w,
    v,
):
    """
    Idaji, M. J., Shamsollahi, M. B.brouillon, & Sardouie, S. H. (2017).
    Higher order spectral regression discriminant analysis (HOSRDA): A tensor
    feature reduction method for ERP detection. Pattern Recognition, 70, 152-162.
    """
    raise NotImplementedError


OBJECTIVES = dict(
    rt=obj_rt,
    tr=obj_tr,
    lfl=obj_lfl,
    od=obj_od,
    sr=obj_sr,
)


def norm_fro(A):
    return tl.sqrt(tl.sum(A**2))


def f_multiway(X, y, classes=None, class_counts=None, method="tr"):
    n_samples, *shape = X.shape
    X = tl.tensor(X, dtype=X.dtype)
    if classes is None or class_counts is None:
        classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    # Calculate class means, overall class mean and center data
    means, X_centered = center(X, y, classes)
    class_mean = tl.mean(means, axis=0)
    if method == "tr":
        tr_scatter_w = norm_fro(X_centered) ** 2
        tr_scatter_b = 0
        for ci, c in enumerate(classes):
            mean_centered = means[ci] - class_mean
            tr_scatter_b += class_counts[ci] * norm_fro(mean_centered) ** 2
        # Calculate Fisher ratio
        F = tr_scatter_b / tr_scatter_w
    elif method == "rt":
        X_centered_flat = tl.unfold(X_centered, 0)

        scatter_w, _ = mode_scatter(
            X_centered_flat, 0, assume_centered=True, shrinkage="lw"
        )
        means_flat = tl.unfold(means, 0)
        scatter_b, _ = mode_scatter(
            means_flat, 0, assume_centered=False, shrinkage="lw"
        )
        _, w = trunc_eigh(scatter_b, scatter_w, rank=None, method="lanczos")
        F = tl.sum(w)
    else:
        raise AttributeError
    return F, 0


def f_oneway(X, y, classes=None, class_counts=None):
    n_samples, *shape = X.shape
    order = len(shape)
    if classes is None or class_counts is None:
        classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    ss_alldata = tl.sum(X**2, axis=0)
    sums_per_class, _ = center(X, y, classes)
    # sums_per_class *= tl.tensor(class_counts)[:, np.newaxis, np.newaxis]
    sums_per_class *= tl.tensor(
        np.expand_dims(class_counts, axis=tuple(np.arange(1, order + 1)))
    )
    square_of_sums_alldata = tl.sum(sums_per_class, axis=0) ** 2
    square_of_sums_per_class = sums_per_class**2
    sstot = ss_alldata - square_of_sums_alldata / n_samples
    ssbn = 0.0
    for ci in range(n_classes):
        ssbn += square_of_sums_per_class[ci] / class_counts[ci]
    ssbn -= square_of_sums_alldata / float(n_samples)
    sswn = sstot - ssbn
    dfbn = n_classes - 1
    dfwn = n_samples - n_classes
    msb = ssbn / dfbn
    msw = sswn / dfwn
    F = msb / msw
    p = backend.scipy.special.fdtrc(dfbn, dfwn, F)
    return F, p


def mode_scatter(
    X, k, weights=None, shrinkage=0, toeplitz=None, taper=False, assume_centered=False
):
    """Calculate the scatter matrix along a given tensor mode"""
    if weights is not None:
        X = (X.T * weights).T
    # elif shrinkage == "oas":
    #    X = tl.unfold(X, k + 1)
    #    scatter, shrinkage = oas(tl.to_numpy(X.T), assume_centered=assume_centered)
    #    scatter = tl.tensor(scatter)
    #    scatter *= X.shape[1] - 1
    if shrinkage == "lw":
        Xf = tl.unfold(X, k + 1).T
        shrinkage = ledoit_wolf_shrinkage(Xf, assume_centered=assume_centered)
    order = len(X.shape[1:])
    modes = [0] + [kk + 1 for kk in range(order) if kk != k]
    if not assume_centered:
        X = X - tl.mean(X, axis=0)
    scatter = tl.tensordot(X, X.conj(), axes=(modes, modes))
    structured = tl.mean(tl.diag(scatter)) * tl.eye(scatter.shape[0], dtype=X.dtype)
    scatter = (1 - shrinkage) * scatter + shrinkage * structured
    if toeplitz is not None and k in toeplitz:
        scatter = force_toeplitz(scatter, taper=taper)

    return scatter, shrinkage


def force_toeplitz(A, taper=False):
    n, _ = A.shape
    toep = tl.zeros(n, dtype=A.dtype)
    for i in range(n):
        diag = tl.diag(A, k=i)
        toep[i] = tl.mean(diag)
    if taper:
        taper = tl.arange(len(toep), 0, -1) - 1
        toep = toep * taper
    return backend.scipy.linalg.toeplitz(toep)


def mse(A, B):
    return tl.abs(tl.mean((B - A) ** 2))


def bic(n, k, ll):
    return tl.log(n) * k - 2 * ll


def aic(k, ll):
    return 2 * k - 2 * ll


def trunc_eigh(
    A,
    B=None,
    rank=None,
    largest=True,
    method="lanczos",
    flip_sign=True,
    **solver_params,
):
    """

    SVD eigensolver can only be used if  B^-1@A is semi-positive definite
    """
    solver_params = solver_params or dict()
    if method == "lanczos":
        v, w = backend.scipy.lanczos(
            A, B=B, rank=rank, largest=largest, **solver_params
        )
    elif method == "svd":
        solver_params["flip_sign"] = flip_sign
        solver_params.setdefault("method", "truncated_svd")
        if largest:
            solver_params["n_eigenvecs"] = rank
        else:
            solver_params["n_eigenvecs"] = None
        if B is None:
            v, w, _ = tl.tenalg.svd_interface(A, **solver_params)
        else:
            v, w, _ = tl.tenalg.svd_interface(
                backend.np.linalg.solve(B @ A), **solver_params
            )
        if not largest:
            w = w[-rank:]
            v = v[:, -rank:]

    elif method == "lobpcg":
        init = solver_params.pop("init", tl.eye(A.shape[0], dtype=A.dtype))
        init = init[:, :rank]
        v, w = backend.scipy.sparse.linalg.lobpcg(
            A, init, B=B, rank=rank, largest=largest, **solver_params
        )

    if flip_sign:
        sign = tl.sign(v[0, :])
        v = v * sign[np.newaxis, :]
    return v, w


class HODA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(
        self,
        max_iter=256,
        tol=1e-8,
        rank=None,
        init="svd",
        shrinkage="lw",
        toeplitz=None,
        taper=False,
        obj="rt",
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
        # X = tl.tensor(X.copy(), dtype=X.dtype)
        X = tl.tensor(X, dtype=X.dtype)
        self.classes_, class_counts = np.unique(y, return_counts=True)
        class_counts = tl.tensor(class_counts)
        n_samples, *shape = X.shape
        order = len(shape)

        # Initialize solver
        if self.obj not in OBJECTIVES.keys():
            raise ValueError(f"objective must be one of {list(OBJECTIVES.keys())}")
        solver_params = self.solver_params
        if solver_params is None:
            solver_params = dict()

        # Calculate means and center
        self.means_, X_centered = center(X, y, self.classes_)
        means_centered = self.means_ - np.mean(self.means_, axis=0)

        # Initialize projections
        if self.verbose:
            print("Initializing factors...")
        self.scalings_ = self._init(X, self.rank)
        scatter_X = [None] * order
        for k in range(order):
            Xk = tl.unfold(X, k + 1)
            scatter_X[k] = Xk @ Xk.T

        # Initialize iterative algorithm
        self.scatter_w_ = [None] * order
        self.scatter_b_ = [None] * order

        # Initialze training information
        self.train_info_ = []
        self.mode_train_info_ = []

        # Iteratively find projections
        if self.verbose:
            print("Fitting discriminative Tucker model...")

        iterator = range(self.max_iter)
        if self.verbose:
            iterator = tqdm(iterator)
        for self.iter_ in iterator:
            new_scalings = [None] * order

            mode_rows = [dict(iteration=self.iter_, mode=k) for k in range(order)]
            for k in range(order):
                modes = range(1, order + 1)
                X_centered_proj = tl.tenalg.multi_mode_dot(
                    X_centered,
                    self.scalings_,
                    modes=modes,
                    skip=k,
                    transpose=True,
                )

                if isinstance(self.shrinkage, tuple):
                    shrinkage = self.shrinkage[k]
                else:
                    shrinkage = self.shrinkage

                scatter_w, shrinkage = mode_scatter(
                    X_centered_proj,
                    k,
                    assume_centered=True,
                    shrinkage=shrinkage,
                    toeplitz=self.toeplitz,
                    taper=self.taper,
                )
                self.scatter_w_[k] = scatter_w

                # Calculate between class scatter
                means_centered_proj = tl.tenalg.multi_mode_dot(
                    means_centered, self.scalings_, modes=modes, skip=k, transpose=True
                )
                scatter_b, _ = mode_scatter(
                    means_centered_proj, k, weights=tl.sqrt(class_counts), shrinkage=0
                )
                self.scatter_b_[k] = scatter_b

                # Solve
                A, B, largest = OBJECTIVES[self.obj](
                    scatter_b, scatter_w, self.scalings_[k]
                )
                if self.solver == "lobpcg":
                    solver_params["init"] = self.scalings_[k]
                u, w = trunc_eigh(
                    A,
                    B,
                    rank=self.rank_(k),
                    method=self.solver,
                    largest=largest,
                    **solver_params,
                )
                u, w = trunc_eigh(
                    u @ u.T @ (scatter_X[k]) @ u @ u.T,
                    rank=self.rank_(k),
                    method="lanczos",
                    largest=largest,
                    **solver_params,
                )
                # Orthonormalize
                u, _ = tl.qr(u, mode="reduced")
                # Flip signs
                # sign = tl.sign(u[0, :])
                # u = u * sign[np.newaxis, :]

                # Store mode training information
                if self.keep_train_info:
                    obj = tl.sum(w)
                    mode_rows[k]["objective"] = obj
                    mode_rows[k]["shrinkage"] = shrinkage
                    mode_rows[k]["rank"] = self.rank_(k)
                new_scalings[k] = u

            # Update scalings
            old_scalings = self.scalings_
            self.scalings_ = new_scalings

            # Calculate update
            update = 0
            for k in range(order):
                u_old = old_scalings[k]
                u_new = self.scalings_[k]
                if u_old.shape != u_new.shape:
                    mode_update = np.inf
                else:
                    mode_update = tl.mean((u_old - u_new) ** 2)
                update += np.log(mode_update) / order
                mode_rows[k]["update"] = mode_update
            update = np.exp(update)
            Xt = self.transform(X)

            # Store iteration training information
            info_row = dict()
            if self.keep_train_info:
                info_row = self._calc_train_info(X, Xt, y, class_counts)
            info_row["iteration"] = self.iter_
            info_row["update"] = update
            self.train_info_.append(info_row)

            self.mode_train_info_ += mode_rows

            ## Check convergence
            if update < self.tol:
                break
        self._fit_forward(X, Xt, y)
        # Convert train_info to dataframe
        self.train_info_ = pd.DataFrame(self.train_info_).astype(float)
        self.train_info_.set_index(["iteration"], inplace=True)
        self.mode_train_info_ = pd.DataFrame(self.mode_train_info_).astype(float)
        self.mode_train_info_.set_index(["iteration", "mode"], inplace=True)
        if self.verbose:
            print(f"Fitted Tucker model of rank {self.ml_rank_} ...")

        return self

    def rank_(self, k):
        return self.scalings_[k].shape[-1]

    @property
    def ml_rank_(self):
        order = len(self.scalings_)
        return tuple([self.rank_(k) for k in range(order)])

    def _init(self, X, rank):
        _, *shape = X.shape
        order = len(shape)
        if rank is None:
            rank = [min(shape) for _ in range(order)]
        elif isinstance(self.rank, int):
            rank = [self.rank for _ in range(order)]

        for k in range(order):
            if rank[k] > shape[k]:
                raise ValueError(
                    f"rank {rank} is not smaller or equal than shape {shape}"
                )

        scalings = [None] * order
        if self.init == "mlsvd":
            modes = tuple(range(1, order + 1))
            (_, scalings), _ = tl.decomposition.partial_tucker(
                X,
                rank=rank,
                modes=modes,
            )
        else:
            for k in range(order):
                if self.init == "identity":
                    scalings[k] = tl.eye(shape[k], rank[k], dtype=X.dtype)
                elif self.init == "ones":
                    scalings[k] = tl.ones((shape[k], rank[k]), dtype=X.dtype)
                elif self.init == "random":
                    scalings[k] = tl_random.random_tensor(
                        shape=(shape[k], rank[k]),
                    )
                    scalings[k], _ = tl.qr(scalings[k], mode="reduced")
                elif self.init == "eye":
                    scalings[k] = tl.eye(shape[k], dtype=X.dtype)[:, : rank[k]]
                elif self.init == "svd":
                    Xk = tl.unfold(X, k + 1)
                    scalings[k], _, _ = tensorly.tenalg.svd_interface(
                        Xk, n_eigenvecs=rank[k]
                    )
                else:
                    raise ValueError(
                        "init should be one of {identity, ones, random, svd}"
                    )
        return scalings

    def _fit_forward(self, X, Xt, y):
        n_samples, *shape = X.shape
        order = len(shape)
        _, X_centered = center(X, y, self.classes_)
        _, Xt_centered = center(Xt, y, self.classes_)

        # self.cov_cross_ = tl.tenalg.tensordot(X_centered, Xt, modes=(0,))
        # self.cov_cross_ /= n_samples - 1
        # (self.cov_cross_), _ = tl.decomposition.partial_tucker(
        #    self.cov_cross_, rank=self.ml_rank_, modes=[k for k in range(order)]
        # )

        # cov_Xt = tl.tenalg.tensordot(Xt_centered, Xt_centered, modes=(0,))
        # cov_Xt /= n_samples - 1
        # cov_Xt_r = cov_Xt.reshape(math.prod(self.ml_rank_), math.prod(self.ml_rank_))
        # cov_Xt_r_inv = backend.np.linalg.pinv(cov_Xt_r)
        # self.cov_Xt_inv_ = tl.reshape(cov_Xt_r_inv, (*self.ml_rank_, *self.ml_rank_))

        # self.A_ = tl.tenalg.tensordot(
        #    self.cov_cross_, self.cov_Xt_inv_, modes=[(2, 3), (0, 1)]
        # )

        self.aps_ = [None] * order
        for k in range(order):
            cov_x, _ = mode_scatter(
                X_centered,
                k,
                shrinkage=self.shrinkage,
                assume_centered=True,
                toeplitz=self.toeplitz,
                taper=self.taper,
            )
            cov_g, _ = mode_scatter(
                Xt_centered, k, shrinkage=self.shrinkage, assume_centered=True
            )
            if k == 0:
                cov_x /= tl.trace(cov_x) / shape[k]
                cov_g /= tl.trace(cov_g) / self.ml_rank_[k]
            else:
                cov_x /= n_samples * math.prod(shape) / shape[k] - 1
                cov_g /= n_samples * math.prod(self.ml_rank_) / self.rank_(k) - 1
            self.aps_[k] = cov_x @ tl.solve(cov_g, self.scalings_[k].T).T

    def _calc_train_info(self, X, Xt, y, class_counts):
        row = dict()
        self._fit_forward(X, Xt, y)
        F_tr, _ = f_multiway(
            Xt,
            y,
            self.classes_,
            class_counts,
            method="tr",
        )
        row["F_tr"] = F_tr

        F_rt, _ = f_multiway(
            Xt,
            y,
            self.classes_,
            class_counts,
            method="rt",
        )

        row["F_rt"] = F_rt
        X_rec = self.inv_transform(Xt)
        row["mse"] = mse(X, X_rec)
        return row

    def transform(self, X, y=None):
        if not tl.is_tensor(X):
            X = tl.tensor(X, dtype=X.dtype)
        order = len(X.shape) - 1
        Xt = tl.tenalg.multi_mode_dot(
            X, self.scalings_, modes=range(1, order + 1), transpose=True
        )
        return Xt

    def inv_transform(self, Xt, y=None):
        order = len(Xt.shape) - 1
        if not tl.is_tensor(Xt):
            Xt = tl.tensor(Xt, dtype=Xt.dtype)
        return tl.tenalg.multi_mode_dot(Xt, self.aps_, modes=[1, 2])

        # return tl.tenalg.tensordot(Xt, self.A_, modes=[(1, 2), (2, 3)])

        # Xt_uncorr = tl.tenalg.tensordot(Xt, self.cov_Xt_inv_, modes=[(1, 2), (0, 1)])
        # cov_cross_core, cov_cross_factors = self.cov_cross_
        # X = tl.tenalg.multi_mode_dot(
        #    tl.tenalg.tensordot(Xt_uncorr, cov_cross_core, modes=[(1, 2), (2, 3)]),
        #    cov_cross_factors,
        #    modes=[1, 2],
        # )
        # return X

    def log_likelihood(X):
        """Based on the Multilinear normal distribution and LDA

        Ohlson, Martin, M. Rauf Ahmad, and Dietrich Von Rosen.
        "The multilinear normal distribution: Introduction and some basic properties."
        Journal of Multivariate Analysis 113 (2013): 37-47.
        """
        raise NotImplemented


class BTTDA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_blocks=8,
        info_crit="bic",
        hoda_params=None,
        keep_train_info=False,
        verbose=False,
    ):
        self.max_blocks = max_blocks
        self.hoda_params = hoda_params
        self.info_crit = info_crit
        self.verbose = verbose
        self.keep_train_info = keep_train_info

    def fit(self, X, y=None):
        X = tl.tensor(X.copy())
        n_samples, *shape = X.shape
        hoda_params = self.hoda_params or dict()
        self.blocks_ = []
        self.train_info_ = []

        # last_crit = np.inf
        crit = np.inf
        err = X.copy()

        for b in range(1, self.max_blocks + 1):
            if self.verbose:
                print(f"Fitting block {b}/{self.max_blocks}...")

            if self.info_crit is not None:
                block = None
                if b > 1:
                    Xt = self.transform(X)
                else:
                    Xt = None
                for r in range(1, min(shape) + 1):
                    # for r in [1]:
                    hoda_params["rank"] = r
                    new_block = HODA(**hoda_params)
                    new_block.fit(err, y)
                    new_Xtb = new_block.transform(err)
                    new_crit = evaluate_info_crit(
                        new_Xtb, y, Xt_extra=Xt, info_crit=self.info_crit
                    )
                    if new_crit >= crit:
                        break
                    block = new_block
                    crit = new_crit
                if block is None:
                    break
            else:
                block = HODA(**hoda_params)
                block.fit(err, y)

            self.blocks_.append(block)
            err -= block.inv_transform(block.transform(err))

            if self.keep_train_info:
                row = dict(block=self.n_blocks_)
                row[self.info_crit] = float(crit)
                row["rank"] = block.ml_rank_
                row["mse"] = float(tl.abs(tl.mean((err) ** 2)))
                self.train_info_.append(row)

        if self.keep_train_info:
            self.train_info_ = pd.DataFrame(self.train_info_)
            self.train_info_.set_index(["block"], inplace=True)

        return self

    @property
    def n_blocks_(self):
        return len(self.blocks_)

    def transform(self, X, y=None, n_blocks=None):
        X = tl.tensor(X.copy(), dtype=X.dtype)
        n_samples, *_ = X.shape
        Xt = []
        if n_blocks is None:
            n_blocks = self.n_blocks_
        for b in range(n_blocks):
            block = self.blocks_[b]
            Xtb = block.transform(X, y)
            Xt.append(Xtb.reshape(n_samples, -1))
            X -= block.inv_transform(Xtb)

        Xt = tl.concatenate(Xt, axis=1)
        return Xt

    def inv_transform(self, Xt, y=None, n_blocks=None):
        n_samples, _ = Xt.shape
        if n_blocks is None:
            n_blocks = self.n_blocks_
        n_blocks = min(n_blocks, self.n_blocks_)
        shape = (n_samples, *[s.shape[0] for s in self.blocks_[0].scalings_])
        X = tl.zeros(shape)
        for b in range(n_blocks):
            block = self.blocks_[b]
            n_features = math.prod(block.ml_rank_)
            Xtb = Xt[:, :n_features]
            Xtb = Xtb.reshape((n_samples, *block.ml_rank_))
            X += block.inv_transform(Xtb)
            Xt = Xt[:, n_features:]
        return X


def evaluate_info_crit(Xt, y, Xt_extra=None, info_crit="bic"):
    n_samples, *_ = Xt.shape
    Xt = Xt.reshape((n_samples, -1))
    if Xt_extra is not None:
        Xt = backend.np.hstack([Xt, Xt_extra])
    Xt = tl.to_numpy(Xt)

    clf = LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")
    clf.fit(Xt, y)
    proba_pred = clf.predict_proba(Xt)
    log_likelihood = -log_loss(y, proba_pred, normalize=False)
    n_features = Xt.shape[-1]
    # Determine information criterion
    if info_crit == "bic":
        value = bic(n_samples, n_features, log_likelihood)
    elif info_crit == "aic":
        value = aic(n_features, log_likelihood)
    else:
        raise NotImplementedError
    return value
