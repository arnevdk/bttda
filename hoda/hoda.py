import math

import cupy
import ipdb
import numpy as np
import scipy.stats
import statsmodels.api as sm
import tensorly as tl
import tensorly.decomposition
import tensorly.tenalg
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.covariance import ledoit_wolf, oas, shrunk_covariance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import ElasticNet
from sklearn.metrics import log_loss
from statsmodels.multivariate.manova import MANOVA
from tensorly import random as tl_random
from tqdm.notebook import tqdm

from hoda.tenalg import (det, force_toeplitz, ledoit_wolf_shrinkage, maximum,
                         solve, trunc_eigh)


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


def aic(X, y_true):
    n_samples = len(y_true)
    _, y_true = np.unique(y_true, return_inverse=True)
    X_flat = X.reshape((n_samples, -1))
    _, n_params = X_flat.shape
    X_flat = cupy.asnumpy(X_flat)
    ols = sm.OLS(y_true, X_flat)
    res = ols.fit()
    # lda = LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")
    # lda.fit(X_flat, y_true)
    # y_pred = lda.predict(X_flat)
    # log_likelihood = -log_loss(y_true, y_pred)
    # aic = 2 * n_params - 2 * log_likelihood
    return res.aic


def f_stat(X, y):
    classes = np.unique(y)
    n_samples, *_ = X.shape
    X_flat = X.reshape((n_samples, -1))
    if X_flat.shape[-1] == 1:
        meas = []
        for c in classes:
            meas.append(cupy.asnumpy(X_flat[y == c]))
        f_stat = float(scipy.stats.f_oneway(*meas).statistic)
    else:
        _, y_int = np.unique(y, return_inverse=True)
        anova = MANOVA(cupy.asnumpy(X_flat), y_int)
        res = anova.mv_test()
        f_stat = res.results["x0"]["stat"].values[0, 3]
    return f_stat


# def fisher_score(X, y):
#    X = tl.tensor(X, dtype=X.dtype)
#    n_samples, *shape = X.shape
#    classes, class_counts = np.unique(y, return_counts=True)
#
#    scatter_t = tl.norm(X, order=2) ** 2
#    # Calculate class means, overall class mean and center data
#    means, X_centered = center(X, y)
#    class_mean = tl.mean(means, axis=0)
#
#    # Calculate between class scatter
#    scatter_b = 0
#    for ci, c in enumerate(classes):
#        scatter_b += class_counts[ci] * tl.norm(means[ci] - class_mean, order=2) ** 2
#    # Calculate f score
#    return float(scatter_b / (scatter_t - scatter_b))
#


def center(X, y):
    _, *shape = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    means = tl.zeros((n_classes, *shape), dtype=X.dtype)
    if tl.get_backend() == "cupy":
        X_centered = tl.zeros((n_classes, *X.shape))
        for ci, c in enumerate(classes):
            X_where = cupy.where(
                cupy.array(y == c)[:, np.newaxis, np.newaxis],
                X,
                cupy.full_like(X, cupy.nan),
            )
            means[ci] = cupy.nanmean(X_where, axis=0)
            X_centered[ci] = X_where - means[ci]
        X_centered = cupy.nansum(X_centered, axis=0)
    else:
        X_centered = []
        for ci, c in enumerate(classes):
            X_where = X[y == c]
            means[ci] = tl.mean(X_where, axis=0)
            X_centered.append(X_where - means[ci])
        X_centered = tl.concatenate(X_centered, axis=0)
    return means, X_centered


def mode_scatter(X, k, weights=None, shrinkage=0, assume_centered=False):
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
        shrinkage = ledoit_wolf_shrinkage(Xf)
    order = len(X.shape[1:])
    modes = [0] + [kk + 1 for kk in range(order) if kk != k]
    if not assume_centered:
        X = X - tl.mean(X, axis=0)
    scatter = tl.tensordot(X, X.conj(), axes=(modes, modes))
    scatter = (1 - shrinkage) * scatter + shrinkage * tl.mean(
        tl.diag(scatter)
    ) * tl.eye(scatter.shape[0], dtype=X.dtype)
    return scatter, shrinkage


def mse(A, B):
    return float(tl.abs(tl.mean((B - A) ** 2)))


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
            self.rank = shape.copy()
        else:
            self.rank_ = self.rank.copy()

        # Initialize solver
        if self.obj not in OBJECTIVES.keys():
            raise ValueError(f"objective must be one of {list(OBJECTIVES.keys())}")
        solver_params = self.solver_params
        if solver_params is None:
            solver_params = dict()

        # Calculate means and center
        means, X_centered = center(X, y)
        means_centered = means - np.mean(means, axis=0)

        # Initialize projections
        if self.verbose:
            print("Initializing factors...")
        self.scalings_ = self._init(X, shape)

        # Initialize iterative algorithm
        for k in range(order):
            self.scalings_[k] = self.scalings_[k][:, : self.rank_[k]]
        self.scatter_w_ = [None] * order
        self.scatter_b_ = [None] * order
        self.scatter_t_ = []
        for k in range(order):
            self.scatter_t_.append(mode_scatter(X, k, assume_centered=False)[0])
        self.lambda_ = 0

        # Initialze training information
        if self.keep_train_info:
            self.train_info_ = dict(
                mode_objective=[[] for _ in range(order)],
                mode_update=[[] for _ in range(order)],
                mode_shrinkage=[[] for _ in range(order)],
                f_stat=[],
                mse=[],
                lambd=[],
                update=[],
            )

        # Iteratively find projections
        if self.verbose:
            print(f"Fitting discriminative tucker model...")

        iterator = range(self.max_iter)
        if self.verbose:
            iterator = tqdm(iterator)
        for self.iter_ in iterator:
            new_scalings = [None] * order
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
                    rank=self.rank[k],
                    method=self.solver,
                    largest=largest,
                    **solver_params,
                )
                # Why this line?
                # u, w, _ = tensorly.tenalg.svd_interface(
                #    u @ u.T @ self.scatter_t_[k] @ u @ u.T,
                #    n_eigenvecs=self.rank_[k],
                # )
                u, _ = tl.qr(u, mode="reduced")

                obj = tl.sum(w)

                # Store mode training information
                if self.keep_train_info:
                    self.train_info_["mode_objective"][k].append(float(obj))
                    self.train_info_["mode_shrinkage"][k].append(float(shrinkage))

                new_scalings[k] = u

            # Update scalings
            old_scalings = self.scalings_
            self.scalings_ = new_scalings
            Xt = self.transform(X)
            self._fit_inverse(X, Xt, y)

            # Lasso
            Xt, self.lambda_ = self._learn_sparse_coef(X, Xt)

            # for k in range(order):
            #    modes = [0] + [kk + 1 for kk in range(order) if kk != k]
            #    Xtk = tl.unfold(Xt, k + 1)
            #    relevance = tl.sum(tl.abs(Xtk), axis=-1)
            #    relevance = relevance / tl.sum(relevance)
            #    keep_idc = relevance > 0.2
            #    self.scalings_[k] = self.scalings_[k][:, keep_idc]
            #    self.rank_[k] = self.scalings_[k].shape[1]

            # Select discriminatory components
            # F = tl.zeros(shape)
            # Xt_class_mean = tl.zeros(shape)
            # for c in self.classes_:
            #    Xt_class_mean += tl.mean(Xt[y == c]) / len(self.classes_)
            # Xt_c = Xt - Xt_class_mean
            # F = tl.mean(tl.abs(Xt_c), axis=0)

            # TODO replace with drop columns
            Xt = self.transform(X)
            self._fit_inverse(X, Xt, y)

            # Calculate update
            update = 0
            for k in range(order):
                u_new = tl.zeros((shape[k], self.rank[k]), dtype=X.dtype)
                u_new[:, : self.scalings_[k].shape[-1]] = self.scalings_[k]
                u_old = tl.zeros((shape[k], self.rank[k]), dtype=X.dtype)
                u_old[:, : old_scalings[k].shape[-1]] = old_scalings[k]
                mode_update = tl.mean((u_old - u_new) ** 2)
                update += mode_update
                if self.keep_train_info:
                    self.train_info_["mode_update"][k].append(mode_update)

            # Store iteration training information
            if self.keep_train_info:
                self.train_info_["f_stat"].append(f_stat(Xt, y))
                X_rec = self.inv_transform(Xt)
                self.train_info_["mse"].append(mse(X, X_rec))
                self.train_info_["lambd"].append(self.lambda_)
                self.train_info_["update"].append(update)

            ## Check convergence
            # if update < self.tol:
            #    break
        if self.verbose:
            print(f"Fitted tucker model of rank {self.rank_} ...")

        return self

    def _init(self, X, rank):
        _, *shape = X.shape
        order = len(shape)
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

    def _fit_inverse(self, X, core, y):
        _, *shape = X.shape
        order = len(shape)
        self.cov_w_ = [None] * order
        self.cov_l_ = []
        self.aps_ = [None] * len(self.scalings_)
        for k in range(order):
            self.cov_l_.append(tl.eye(self.rank_[k]))

        _, core_centered = center(core, y)
        for k in range(order):
            self.cov_w_[k] = self.scatter_w_[k] / (
                math.prod(core.shape) / core.shape[k + 1] - 1
            )
            scatter_l, _ = mode_scatter(core_centered, k, assume_centered=True)
            cov_l = scatter_l / (math.prod(core.shape) / core.shape[k + 1] - 1)
            self.cov_l_[k] = cov_l
            # Haufe method
            self.aps_[k] = self.cov_w_[k] @ solve(self.cov_l_[k], self.scalings_[k].T).T

    def _learn_sparse_coef(self, X, Xt):
        se = tl.sum((X - self.inv_transform(Xt)) ** 2)
        snr = 1
        epsilon = 10 ** (-snr / 10) * tl.sum(X**2)
        tol = 1e-8
        # if se >= epsilon:
        #    return Xt, self.lambda_
        lambda_l = tl.tensor(0)
        lambda_h = tl.max(tl.abs(Xt))
        sign = tl.sign(Xt)
        abs_Xt = tl.abs(Xt)
        zeros = tl.zeros_like(Xt)
        # lambda_h = tl.min(tl.abs(Xt))
        # while True:
        #    Xt_sparse = tl.sign(Xt) * maximum(tl.abs(Xt) - lambda_h, tl.zeros_like(Xt))0
        #    se = tl.sum((X - self.inv_transform(Xt_sparse)) ** 2)
        #    if se >= epsilon:
        #        break
        #    else:
        #        lambda_l = lambda_h
        #        lambda_h = 2 * lambda_h
        tol_iter = int(math.log(1 / tol))
        for i in range(tol_iter):
            lambda_m = (lambda_l + lambda_h) / 2
            Xt_sparse = sign * maximum(abs_Xt - lambda_m, zeros)
            se = tl.sum((X - self.inv_transform(Xt_sparse)) ** 2)
            # Faster than if-statement on GPU
            se_gt_eps = se >= epsilon
            lambda_h = se_gt_eps * lambda_m + (1 - se_gt_eps) * lambda_h
            lambda_l = se_gt_eps * lambda_l + (1 - se_gt_eps) * lambda_m
            # if se >= epsilon:
            #    lambda_h = lambda_m
            # else:
            #    lambda_l = lambda_m
        return Xt_sparse, lambda_m

    def transform(self, X, y=None):
        if not tl.is_tensor(X):
            X = tl.tensor(X, dtype=X.dtype)
        order = len(X.shape) - 1
        Xt = tl.tenalg.multi_mode_dot(
            X, self.scalings_, modes=range(1, order + 1), transpose=True
        )
        return Xt

    def inv_transform(self, Xt, y=None):
        if not tl.is_tensor(Xt):
            Xt = tl.tensor(Xt, dtype=Xt.dtype)
        order = len(Xt.shape) - 1
        X = tl.tenalg.multi_mode_dot(
            Xt, self.aps_, modes=range(1, order + 1), transpose=False
        )
        return X


class BTTDA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_blocks=None,
        hoda_params=None,
        verbose=False,
        keep_train_info=False,
    ):
        self.hoda_params = hoda_params
        self.n_blocks = n_blocks
        self.verbose = verbose
        self.keep_train_info = keep_train_info

    def fit(self, X, y):
        X = tl.tensor(X.copy(), dtype=X.dtype)
        self.classes_, y_num, class_counts = np.unique(
            y, return_inverse=True, return_counts=True
        )
        n_samples = len(y)

        hoda_params = self.hoda_params
        if hoda_params is None:
            hoda_params = dict()

        if self.keep_train_info:
            self.train_info_ = dict(f_stat=[], mse=[], n_params=[], aic=[], bic=[])

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
                self.train_info_["f_stat"].append(f_stat(Xt, y))
                self.train_info_["mse"].append(mse(X, X_rec))
                self.train_info_["n_params"].append(Xt.shape[-1])
                # self.train_info_["aic"].append(aic(Xt, y))
                _, y_num = np.unique(y_num, return_inverse=True)
                Xt_flat = Xt.reshape((n_samples, -1))
                _, n_params = Xt_flat.shape
                X_flat = cupy.asnumpy(Xt_flat)
                ols = sm.OLS(y_num, sm.add_constant(X_flat))
                res = ols.fit()
                print(res.summary())
                self.train_info_["aic"].append(res.aic)
                self.train_info_["bic"].append(res.bic)

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
