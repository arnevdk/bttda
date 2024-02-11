import math

import ipdb
import numpy as np
import pandas as pd
import tensorly as tl
import tensorly.decomposition
import tensorly.tenalg
from numpy.linalg import LinAlgError
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorly import random as tl_random
from tqdm.notebook import tqdm

from hoda.backend import lanczos, lobpcg
from hoda.classification import SelectFweAtLeastOne, Vectorize
from hoda.cov import center, mode_scatter

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
    return F


def trunc_eigh(
    A,
    B=None,
    rank=None,
    largest=True,
    method="lanczos",
    **solver_params,
):
    """

    SVD eigensolver can only be used if  B^-1@A is semi-positive definite
    """
    solver_params = solver_params or dict()
    if method == "lanczos":
        try:
            v, w = lanczos(A, B=B, rank=rank, largest=largest, **solver_params)
        except LinAlgError:
            v, w = v, w = lanczos(
                A, B=B, rank=rank, largest=largest, force_spd=True, **solver_params
            )

    elif method == "svd":
        solver_params.setdefault("method", "truncated_svd")
        if largest:
            solver_params["n_eigenvecs"] = rank
        else:
            solver_params["n_eigenvecs"] = None
        if B is None:
            v, w, _ = tl.tenalg.svd_interface(A, **solver_params)
        else:
            v, w, _ = tl.tenalg.svd_interface(tl.solve(B @ A), **solver_params)
        if not largest:
            w = w[-rank:]
            v = v[:, -rank:]

    elif method == "lobpcg":
        init = solver_params.pop("init", tl.eye(A.shape[0], dtype=A.dtype))
        init = init[:, :rank]
        try:
            w, v = lobpcg(A, init, B=B, largest=largest, **solver_params)
        except (LinAlgError, AttributeError):
            v, w = lanczos(
                A, B=B, rank=rank, largest=largest, force_spd=True, **solver_params
            )

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
        extra_train_info=False,
        random_state=42,
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
        self.extra_train_info = extra_train_info
        self.taper = taper
        self.random_state = random_state

    def fit(self, X, y):
        # Setup
        if not tl.is_tensor(X):
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
        modes = [k + 1 for k in range(order)]
        if isinstance(self.rank, int):
            rank = [self.rank] * order
        elif self.rank is None:
            rank = shape
        else:
            rank = self.rank
        if self.init == "mlsvd":
            (_, self.weights_), _ = tl.decomposition.partial_tucker(
                X_centered,
                rank=rank,
                modes=modes,
            )
        else:
            _, self.weights_ = tl.decomposition._tucker.initialize_tucker(
                X_centered, rank, modes, self.random_state, init=self.init
            )

        # Calculate total scatter
        scatter_x = [None] * order
        for k in range(order):
            scatter_x[k], _ = mode_scatter(
                X,
                k,
                assume_centered=True,
                shrinkage=self.shrinkage,
                toeplitz=self.toeplitz,
                taper=self.taper,
            )

        # Initialize iterative algorithm
        self.scatter_w_ = [None] * order
        self.scatter_b_ = [None] * order

        # Initialze training information
        self.train_info_ = []

        # Iteratively find projections
        if self.verbose:
            print(f"Fitting discriminative Tucker model of rank {self.ml_rank_}...")

        iterator = range(self.max_iter)
        if self.verbose:
            iterator = tqdm(iterator)
        for self.iter_ in iterator:
            converged = True
            for k in range(order):
                modes = range(1, order + 1)
                X_centered_proj = tl.tenalg.multi_mode_dot(
                    X_centered,
                    self.weights_,
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
                    means_centered, self.weights_, modes=modes, skip=k, transpose=True
                )
                scatter_b, _ = mode_scatter(
                    means_centered_proj, k, weights=tl.sqrt(class_counts), shrinkage=0
                )
                self.scatter_b_[k] = scatter_b

                # Solve
                A, B, largest = OBJECTIVES[self.obj](
                    scatter_b, scatter_w, self.weights_[k]
                )
                if self.solver == "lobpcg":
                    solver_params["init"] = self.weights_[k]
                u, w = trunc_eigh(
                    A,
                    B,
                    rank=self.rank_(k),
                    method=self.solver,
                    largest=largest,
                    **solver_params,
                )
                u, w = trunc_eigh(
                    u @ u.T @ (scatter_x[k]) @ u @ u.T,
                    rank=self.rank_(k),
                    method="lanczos",
                    largest=largest,
                    **solver_params,
                )

                # Calculate update and check convergence
                update = tl.metrics.regression.MSE(u, self.weights_[k])
                converged = update < self.tol and converged

                # Store mode training information
                train_info_row = dict()
                train_info_row["iteration"] = self.iter_
                train_info_row["mode"] = k
                train_info_row["update"] = update
                train_info_row["shrinkage"] = shrinkage
                obj = tl.sum(w)
                train_info_row["objective"] = obj
                if self.extra_train_info:
                    train_info_row.update(self._extra_train_info(X, y, class_counts))
                self.train_info_.append(train_info_row)

                # Update weights
                self.weights_[k] = u

            # Exit if converged
            if converged:
                break

        Xt = self.transform(X)
        self._fit_forward(X, Xt, y)

        # Convert train_info to dataframe
        self.train_info_ = pd.DataFrame(self.train_info_).astype(float)

        return self

    def rank_(self, k):
        return self.weights_[k].shape[-1]

    @property
    def ml_rank_(self):
        order = len(self.weights_)
        return tuple([self.rank_(k) for k in range(order)])

    def _fit_forward(self, X, Xt, y):
        n_samples, *shape = X.shape
        order = len(shape)
        _, X_centered = center(X, y, self.classes_)
        _, Xt_centered = center(Xt, y, self.classes_)

        self.cov_x_ = [None] * order
        self.cov_g_ = [None] * order
        self.scale_x_ = tl.zeros(order)
        self.scale_g_ = tl.zeros(order)
        self.aps_ = [None] * order
        for k in range(order):
            cov_x, shrink_x = mode_scatter(
                X_centered,
                k,
                shrinkage=self.shrinkage,
                assume_centered=True,
                toeplitz=self.toeplitz,
                taper=self.taper,
            )
            cov_g, shrink_g = mode_scatter(
                Xt_centered,
                k,
                # shrinkage=self.shrinkage,
                assume_centered=True,
            )

            scale_x = (tl.trace(cov_x) / shape[k]) / (
                n_samples * math.prod(shape) / shape[k] - 1
            )
            scale_g = (tl.trace(cov_g) / self.ml_rank_[k]) / (
                n_samples * math.prod(self.ml_rank_) / self.rank_(k) - 1
            )

            cov_x /= tl.trace(cov_x) / shape[k]
            cov_g /= tl.trace(cov_g) / self.ml_rank_[k]

            self.cov_g_[k] = cov_g
            self.cov_x_[k] = cov_x
            self.scale_x_[k] = scale_x
            self.scale_g_[k] = scale_g
            self.aps_[k] = cov_x @ tl.solve(cov_g, self.weights_[k].T).T
        self.scale_x_ = tl.mean(self.scale_x_)
        self.scale_g_ = tl.mean(self.scale_g_)

    def _extra_train_info(self, X, y, class_counts):
        info = dict()
        n_samples = X.shape[0]
        Xt = self.transform(X)
        self._fit_forward(X, Xt, y)

        # Objective: multi-way F-score
        # trace-ratio
        F_tr, _ = f_multiway(
            Xt,
            y,
            self.classes_,
            class_counts,
            method="tr",
        )
        info["F_tr"] = F_tr
        # ratio-trace
        F_rt, _ = f_multiway(
            Xt,
            y,
            self.classes_,
            class_counts,
            method="rt",
        )
        info["F_rt"] = F_rt

        # MSE
        X_rec = self.inv_transform(Xt)
        info["mse"] = tl.metrics.regression.MSE(X, X_rec)
        # Log-likelihood
        info["log_like"] = log_likelihood(Xt, y)
        # Information criteria
        k = math.prod(self.ml_rank_)
        for crit, func in info_crit.items():
            info[crit] = func(n_samples, k, info["log_like"])

        return info

    def transform(self, X, y=None):
        if not tl.is_tensor(X):
            X = tl.tensor(X, dtype=X.dtype)
        order = len(X.shape) - 1
        Xt = tl.tenalg.multi_mode_dot(
            X, self.weights_, modes=range(1, order + 1), transpose=True
        )
        return Xt

    def inv_transform(self, Xt, y=None):
        if not tl.is_tensor(Xt):
            Xt = tl.tensor(Xt, dtype=Xt.dtype)
        return (
            tl.tenalg.multi_mode_dot(Xt, self.aps_, modes=[1, 2])
            * self.scale_x_
            / self.scale_g_
        )

    @property
    def n_params_(self):
        return sum([s.size for s in self.weights_])


class BTTDA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_blocks=8,
        info_crit="bic",
        hoda_params=None,
        extra_train_info=False,
        verbose=False,
    ):
        self.max_blocks = max_blocks
        self.hoda_params = hoda_params
        self.info_crit = info_crit
        self.verbose = verbose
        self.extra_train_info = extra_train_info

    def fit(self, X, y=None):
        X = tl.tensor(X.copy())
        n_samples, *shape = X.shape
        hoda_params = self.hoda_params or dict()
        self.blocks_ = []
        if self.extra_train_info:
            self.train_info_ = []

        crit_value = np.inf
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

                crit_value = np.inf
                for r in range(1, min(shape) + 1):
                    hoda_params["rank"] = r
                    new_block = HODA(**hoda_params)
                    new_block.fit(err, y)
                    Xtb = new_block.transform(err)
                    Xtb = Xtb.reshape((n_samples, -1))
                    if b > 1:
                        new_Xt = tl.concatenate([Xt, Xtb], axis=-1)
                    else:
                        new_Xt = Xtb
                    new_Xt = Xtb
                    new_log_like = log_likelihood(new_Xt, y)
                    new_n_params = new_Xt.shape[-1]
                    new_crit_value = info_crit[self.info_crit](
                        n_samples, new_n_params, new_log_like
                    )
                    if new_crit_value >= crit_value:
                        break
                    block = new_block
                    crit_value = new_crit_value

                    # if new_crit_value < crit_value
                    #    block = new_block
                    #    crit_value = new_crit_value

                if block is None:
                    break
            else:
                block = HODA(**hoda_params)
                block.fit(err, y)

            self.blocks_.append(block)
            G = block.transform(err)
            err -= block.inv_transform(G)
            if self.extra_train_info:
                row = dict()
                row["block"] = self.n_blocks_
                Xt = self.transform(X)
                row["log_like"] = log_likelihood(Xt, y)
                row["n_params"] = self.n_params_
                k = Xt.shape[-1]
                for crit, func in info_crit.items():
                    row[crit] = float(func(n_samples, k, row["log_like"]))
                row["rank"] = block.ml_rank_
                row["mse"] = float(tl.mean((err) ** 2))
                self.train_info_.append(row)

        if self.extra_train_info:
            self.train_info_ = pd.DataFrame(self.train_info_)
            self.train_info_.set_index(["block"], inplace=True)

        return self

    @property
    def n_blocks_(self):
        return len(self.blocks_)

    @property
    def n_params_(self):
        return sum([b.n_params_ for b in self.blocks_])

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
        shape = (n_samples, *[s.shape[0] for s in self.blocks_[0].weights_])
        X = tl.zeros(shape)
        for b in range(n_blocks):
            block = self.blocks_[b]
            n_features = math.prod(block.ml_rank_)
            Xtb = Xt[:, :n_features]
            Xtb = Xtb.reshape((n_samples, *block.ml_rank_))
            X += block.inv_transform(Xtb)
            Xt = Xt[:, n_features:]
        return X

    @property
    def block_rank_(self):
        return tuple([b.ml_rank_ for b in self.blocks_])


def bic(n, k, log_like):
    return k * tl.log(n) - 2 * log_like


def aic(_, k, log_like):
    return 2 * k - 2 * log_like


def aicc(n, k, log_like):
    return aic(n, k, log_like) + (2 * k**2 + 2 * k) / (n - k - 1)


info_crit = dict(
    bic=bic,
    aic=aic,
    aicc=aicc,
)


def log_likelihood(Xt, y):
    clf = make_pipeline(
        Vectorize(),
        # SelectFweAtLeastOne(),
        LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr"),
    )
    clf.fit(Xt, y)
    proba = clf.predict_proba(Xt)
    return -log_loss(y, proba, normalize=False)
