import math

import numpy as np
import pandas as pd
import tensorly as tl
import tensorly.decomposition
import tensorly.tenalg
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss
from tensorly import random as tl_random
from tqdm.notebook import tqdm

import hoda.backend as backend
from hoda.gpu_opt import center, combine_pvalues, ledoit_wolf_shrinkage

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
    return scatter_b - psi * phi * scatter_w, None, True


def obj_lfl(scatter_b, scatter_w, u, psi=1):
    """Linear feature learning obbjective.

    Aghili, S. N., Kilani, S., Khushaba, R. N., & Rouhani, E. (2023).
    A spatial-temporal linear feature learning algorithm for P300-based -
    brain-computer interfaces. Heliyon, 9(4).
    """
    phi = tl.trace(u.T @ scatter_b @ u) / tl.trace(u.T @ scatter_w @ u)
    return scatter_b - psi * phi * scatter_w, scatter_w, True


def obj_od(scatter_b, scatter_w, v):
    """Optimal dimensionality discriminant analysis

    Nie, F., Xiang, S., Song, Y., & Zhang, C. (2007, April).
    Extracting the optimal dimensionality for discriminant analysis. In 2007
    IEEE International Conference on Acoustics, Speech and Signal Processing-ICASSP'07 (Vol. 2, pp. II-617). IEEE.

    Wang, J., Wang, L., Nie, F., & Li, X. (2021). A novel formulation of trace ratio linear discriminant analysis. IEEE Transactions on Neural Networks and Learning Systems, 33(10), 5568-5578.
    """
    s = tl.trace(v.T @ scatter_b @ v) / tl.trace(v.T @ scatter_w @ v)
    return s**2 * scatter_w - 2 * s * scatter_b, None, False


def obj_sr(
    scatter_b,
    scatter_w,
    v,
):
    """
    Idaji, M. J., Shamsollahi, M. B., & Sardouie, S. H. (2017).
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


def f_multiway(X, y, classes=None, class_counts=None):
    X = tl.tensor(X, dtype=X.dtype)
    n_samples, *shape = X.shape
    if classes is None or class_counts is None:
        classes, class_counts = np.unique(y, return_counts=True)

    # Calculate class means, overall class mean and center data
    means, X_centered = center(X, y, classes)
    class_mean = tl.mean(means, axis=0)
    scatter_w = tl.norm(X_centered, order=2) ** 2
    # Calculate between class scatter
    scatter_b = 0
    for ci, c in enumerate(classes):
        scatter_b += class_counts[ci] * tl.norm(means[ci] - class_mean, order=2) ** 2
    # Calculate f score
    F = float(scatter_b / scatter_w)
    return F


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
    f = msb / msw
    prob = backend.scipy.special.fdtrc(dfbn, dfwn, f)
    return f, prob


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
        shrinkage = ledoit_wolf_shrinkage(Xf, assume_centered=assume_centered)
    order = len(X.shape[1:])
    modes = [0] + [kk + 1 for kk in range(order) if kk != k]
    if not assume_centered:
        X = X - tl.mean(X, axis=0)
    scatter = tl.tensordot(X, X.conj(), axes=(modes, modes))
    structured = tl.mean(tl.diag(scatter)) * tl.eye(scatter.shape[0], dtype=X.dtype)
    scatter = (1 - shrinkage) * scatter + shrinkage * structured
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
    return float(tl.abs(tl.mean((B - A) ** 2)))


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
        max_iter=100,
        tol=1e-12,
        rank=None,
        init="svd",
        shrinkage="lw",
        toeplitz=None,
        taper=False,
        obj="rt",
        solver="lanczos",
        lasso=False,
        prune=True,
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
        self.lasso = lasso
        self.prune = prune

    def fit(self, X, y):
        # Setup
        # X = tl.tensor(X.copy(), dtype=X.dtype)
        X = tl.tensor(X, dtype=X.dtype)
        self.classes_, class_counts = np.unique(y, return_counts=True)
        class_counts = tl.tensor(class_counts)
        n_samples, *shape = X.shape
        self.shape_ = shape
        order = len(shape)
        if self.rank is None:
            self.rank_ = shape.copy()
            # self.rank_ = [min(shape) for _ in range(order)]
        else:
            self.rank_ = self.rank.copy()

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
        self.scalings_ = self._init(X, shape)

        # Initialize iterative algorithm
        for k in range(order):
            self.scalings_[k] = self.scalings_[k][:, : self.rank_[k]]
        self.scatter_w_ = [None] * order
        self.scatter_b_ = [None] * order
        self.lambda_ = 0

        # Initialze training information
        if self.keep_train_info:
            self.train_info_ = []
            self.mode_train_info_ = []
        if self.prune:
            snr = tl.tensor(
                f_multiway(X, y), classes=self.classes_, class_counts=class_counts
            )

        # Iteratively find projections
        if self.verbose:
            print("Fitting discriminative Tucker model...")

        iterator = range(self.max_iter)
        if self.verbose:
            iterator = tqdm(iterator)
        for self.iter_ in iterator:
            new_scalings = [None] * order

            if self.keep_train_info:
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
                    X_centered_proj, k, assume_centered=True, shrinkage=shrinkage
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
                if self.solver == "lobpcg":
                    solver_params["init"] = self.scalings_[k]
                u, w = trunc_eigh(
                    A,
                    B,
                    rank=self.rank_[k],
                    method=self.solver,
                    largest=largest,
                    **solver_params,
                )
                u, _ = tl.qr(u, mode="reduced")

                # Store mode training information
                if self.keep_train_info:
                    obj = tl.sum(w)
                    mode_rows[k]["objective"] = float(obj)
                    mode_rows[k][f"shrinkage"] = float(shrinkage)

                new_scalings[k] = u

            # Update scalings
            old_scalings = self.scalings_
            self.scalings_ = new_scalings
            Xt = self.transform(X)

            # Lasso
            if self.lasso:
                self._fit_inverse(X, Xt, y)
                Xt, self.lambda_ = self._learn_sparse_coef(X, Xt, snr=10)
            if self.prune:
                self._prune(Xt, y, snr, self.classes_, class_counts)

            ## Orthonormalize
            for k in range(order):
                self.scalings_[k], _ = tl.qr(self.scalings_[k], mode="reduced")
            Xt = self.transform(X)

            # Calculate update
            update = 0
            for k in range(order):
                u_new = tl.zeros((shape[k], shape[k]), dtype=X.dtype)
                u_new[:, : self.scalings_[k].shape[-1]] = self.scalings_[k]
                u_old = tl.zeros((shape[k], shape[k]), dtype=X.dtype)
                u_old[:, : old_scalings[k].shape[-1]] = old_scalings[k]
                mode_update = tl.mean((u_old - u_new) ** 2)
                update += np.log(mode_update) / order
                if self.keep_train_info:
                    mode_rows[k]["update"] = float(tl.to_numpy(mode_update))
                    mode_rows[k]["rank"] = tl.to_numpy(self.rank_[k])
            update = np.exp(update)

            # Store iteration training information
            if self.keep_train_info:
                self._fit_inverse(X, Xt, y)
                row = dict()
                row["iteration"] = self.iter_
                row["f_stat"] = tl.to_numpy(
                    f_multiway(Xt, y, self.classes_, class_counts)
                )
                X_rec = self.inv_transform(Xt)
                row["mse"] = tl.to_numpy(mse(X, X_rec))
                row["lambd"] = tl.to_numpy(self.lambda_)
                row["update"] = float(tl.to_numpy(update))
                self.train_info_.append(row)
                self.mode_train_info_ += mode_rows

            ## Check convergence
            if update < self.tol:
                break
        self._fit_inverse(X, Xt, y)
        self._sort_components(Xt, y, self.classes_, class_counts)
        if self.keep_train_info:
            self.train_info_ = pd.DataFrame(self.train_info_)
            self.train_info_.set_index(["iteration"], inplace=True)
            self.mode_train_info_ = pd.DataFrame(self.mode_train_info_)
            self.mode_train_info_.set_index(["iteration", "mode"], inplace=True)
        if self.verbose:
            print(f"Fitted Tucker model of rank {self.rank_} ...")

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

        _, core_centered = center(core, y, self.classes_)
        for k in range(order):
            self.cov_w_[k] = self.scatter_w_[k] / (
                math.prod(core.shape) / core.shape[k + 1] - 1
            )
            scatter_l, _ = mode_scatter(core_centered, k, assume_centered=True)
            cov_l = scatter_l / (math.prod(core.shape) / core.shape[k + 1] - 1)
            self.cov_l_[k] = cov_l
            # Haufe method
            self.aps_[k] = (
                self.cov_w_[k]
                @ backend.np.linalg.solve(self.cov_l_[k], self.scalings_[k].T).T
            )

    def _learn_sparse_coef(self, X, Xt, snr=5):
        se = tl.sum((X - self.inv_transform(Xt)) ** 2)
        epsilon = 10 ** (-snr / 10) * tl.sum(X**2)
        tol = 1e-4
        lambda_l = tl.min(tl.abs(Xt))
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
        tol_iter = int(math.log(1 / tol, 2))
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

    def _prune(self, Xt, y, snr, classes=None, class_counts=None):
        n_samples, *shape = Xt.shape
        order = len(Xt.shape) - 1
        for k in range(order):
            mode_F = tl.zeros(shape[k], dtype=Xt.dtype)
            for r in range(shape[k]):
                mode_slice = np.take(Xt, r, axis=k + 1)
                mode_F[r] = f_multiway(
                    mode_slice,
                    y,
                    classes=classes,
                    class_counts=class_counts,
                )
            idc = mode_F > snr
            if not np.any(idc):
                idc = [np.argmax(mode_F)]
            self.scalings_[k] = self.scalings_[k][:, idc]
            self.rank_[k] = self.scalings_[k].shape[1]

    def _sort_components(self, Xt, y, classes=None, class_counts=None):
        n_samples, *shape = Xt.shape
        order = len(Xt.shape) - 1
        for k in range(order):
            mode_F = tl.zeros(shape[k], dtype=Xt.dtype)
            for r in range(shape[k]):
                mode_F[r] = f_multiway(
                    np.take(Xt, r, axis=k + 1),
                    y,
                    classes=classes,
                    class_counts=class_counts,
                )
            sort_idc = np.argsort(-mode_F)
            self.scalings_[k] = self.scalings_[k][:, sort_idc]
            self.aps_[k] = self.aps_[k][:, sort_idc]

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
        n_blocks=None,
        hoda_params=None,
        deflate_transform=False,
        verbose=False,
        keep_train_info=False,
    ):
        self.hoda_params = hoda_params
        self.n_blocks = n_blocks
        self.verbose = verbose
        self.keep_train_info = keep_train_info
        self.deflate_transform = deflate_transform

    def fit(self, X, y):
        X = tl.tensor(X, dtype=X.dtype)
        _, *shape = X.shape
        self.classes_, y_num, class_counts = np.unique(
            y, return_inverse=True, return_counts=True
        )

        hoda_params = self.hoda_params
        if hoda_params is None:
            hoda_params = dict()

        if self.keep_train_info:
            self.train_info_ = []

        self.blocks_ = []
        n_blocks = self.n_blocks or 16
        X_rec = np.zeros_like(X)
        X_defl = X.copy()
        norm_X = tl.norm(X)
        # Deflation scheme
        for self.n_blocks_ in range(1, n_blocks + 1):
            if self.verbose:
                print(f"Fitting block {self.n_blocks_}/{self.n_blocks}...")
            # Fit Tucker block
            block = HODA(**hoda_params)
            block.fit(X_defl, y)
            self.blocks_.append(block)
            # Reconstruct and subtract for next iteration
            Xt_block = block.transform(X_defl)
            X_approx = block.inv_transform(Xt_block)
            X_defl -= X_approx
            X_rec += X_approx
            explained = tl.norm(X_rec) / norm_X
            if self.verbose:
                print()
            if self.keep_train_info:
                row = dict()
                row["block"] = self.n_blocks_ - 1
                row["mse"] = tl.to_numpy(mse(X, X_rec))
                row["explained"] = float(explained)
                Xt_block = block.transform(X)
                row["f_stat"] = tl.to_numpy(f_multiway(Xt_block, y))
                self.train_info_.append(row)
        if self.keep_train_info:
            self.train_info_ = pd.DataFrame(self.train_info_)
            self.train_info_.set_index(["block"], inplace=True)
        return self

    def transform(self, X, y=None, n_blocks=None):
        X = tl.tensor(X.copy(), dtype=X.dtype)
        n_samples, *_ = X.shape
        Xt = []
        if n_blocks is None:
            n_blocks = self.n_blocks_
        for b in range(n_blocks):
            block = self.blocks_[b]
            Xtb = block.transform(X, y)
            if self.deflate_transform:
                X -= block.inv_transform(Xtb)
            Xt.append(Xtb.reshape(n_samples, -1))
        Xt = tl.concatenate(Xt, axis=1)
        return Xt

    def n_features(self, n_blocks=None, mode=None):
        if n_blocks is None:
            n_blocks = len(self.blocks_)
        if mode is None:
            return sum([math.prod(b.rank_) for b in self.blocks_[:n_blocks]])
        else:
            return sum([b.rank_[mode] for b in self.blocks_[:n_blocks]])
