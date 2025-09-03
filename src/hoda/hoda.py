import math
import os
import pdb
import warnings

import numpy as np
import scipy.linalg
import tensorly as tl
import tensorly.decomposition
from numpy.linalg import LinAlgError
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import get_scorer, log_loss
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from hoda.cov import mode_scatter
from hoda.tensorize import Vectorize, vec
from hoda.util import center, flip_signs, ridge_regression, solve_gevdh

# from statsmodels.discrete.discrete_model import Logit
# from tqdm.notebook import tqdm


def obj_rt(scatter_b, scatter_w, _):
    """Ratio trace objective Tr(uT Sw^-1 Sb u).

    Phan, A. H., & Cichocki, A. (2010).
    Tensor decompositions for feature extraction and classification of high
    dimensional datasets. Nonlinear theory and its applications, IEICE, 1(1), 37-68.

    Wang, H., Yan, S., Xu, D., Tang, X., & Huang, T. (2007, June). Trace ratio
    vs. ratio trace for dimensionality reduction. In 2007 IEEE Conference on
    Computer Vision and Pattern Recognition (pp. 1-8). IEEE.

    """
    return scatter_b, scatter_w


def obj_tr(scatter_b, scatter_w, u):
    """Trace ratio objective Tr(uT Sb u)/Tr(uT Sw u).

    Phan, A. H., & Cichocki, A. (2010).
    Tensor decompositions for feature extraction and classification of high
    dimensional datasets. Nonlinear theory and its applications, IEICE, 1(1), 37-68.

    Wang, H., Yan, S., Xu, D., Tang, X., & Huang, T. (2007, June). Trace ratio
    vs. ratio trace for dimensionality reduction. In 2007 IEEE Conference on
    Computer Vision and Pattern Recognition (pp. 1-8). IEEE.
    """
    scatter_b_t = u.T @ scatter_b @ u
    tr_scatter_b_t = tl.trace(scatter_b_t)
    scatter_w_t = u.T @ scatter_w @ u
    tr_scatter_w_t = tl.trace(scatter_w_t)
    phi = tr_scatter_b_t / tr_scatter_w_t
    A = scatter_b - phi * scatter_w
    return A, None


def obj_lfl(scatter_b, scatter_w, u, psi=1):
    """Linear feature learning obbjective.

    Aghili, S. N., Kilani, S., Khushaba, R. N., & Rouhani, E. (2023).
    A spatial-temporal linear feature learning algorithm for P300-based -
    brain-computer interfaces. Heliyon, 9(4).
    """
    phi = tl.trace(u.T @ scatter_b @ u) / tl.trace(u.T @ scatter_w @ u)
    A = scatter_b - phi * scatter_w
    B = scatter_w
    return A, B


def obj_od(scatter_b, scatter_w, u):
    """Optimal dimensionality discriminant analysis

    Nie, F., Xiang, S., Song, Y., & Zhang, C. (2007, April).
    Extracting the optimal dimensionality for discriminant analysis. In 2007
    IEEE International Conference on Acoustics, Speech and Signal Processing-ICASSP'07 (Vol. 2, pp. II-617). IEEE.

    Wang, J., Wang, L., Nie, F., & Li, X. (2021). A novel formulation of trace ratio linear discriminant analysis. IEEE Transactions on Neural Networks and Learning Systems, 33(10), 5568-5578.
    """
    s = tl.trace(u.T @ scatter_b @ u) / tl.trace(u.T @ scatter_w @ u)
    return -(s**2 * scatter_w - 2 * s * scatter_b), None


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


def validate(X, y=None):
    tl.initialize_backend()
    tl.tenalg.set_backend("einsum")
    tl.plugins.use_opt_einsum()

    if not tl.is_tensor(X):
        X = tl.tensor(X)
    
    return X, y


class HODA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(
        self,
        max_iter=256,
        tol=1e-6,
        rank=None,
        shrinkage="lw",
        refit_shrinkage=False,
        toeplitz=None,
        taper=False,
        obj="tr",
        solver="lanczos",
        verbose=False,
        solver_params=None,
        extra_train_info=False,
        theta=None,
        forward=False,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.rank = rank
        self.shrinkage = shrinkage
        self.refit_shrinkage = refit_shrinkage
        self.toeplitz = toeplitz
        self.obj = obj
        self.solver = solver
        self.verbose = verbose
        self.solver_params = solver_params
        self.extra_train_info = extra_train_info
        self.taper = taper
        self.theta = theta
        self.forward = forward

    def fit(self, X, y, classes=None, class_counts=None):
        X, y = validate(X, y)
        # Calculate means, centering and classes once (slow on GPU)
        if classes is None or class_counts is None:
            self.classes_, class_counts = np.unique(y, return_counts=True)
            class_counts = tl.tensor(class_counts)
        else:
            self.classes_ = classes
        self.means_, X_centered = center(X, y, self.classes_)

        # Fit backward model
        self.fit_backward(
            X,
            y,
            X_centered=X_centered,
            means=self.means_,
            classes=self.classes_,
            class_counts=class_counts,
        )

        # Fit forward model
        if self.forward:
            self.fit_forward(X, y, X_centered=X_centered)
        return self

    def fit_backward(
        self,
        X,
        y,
        X_centered=None,
        means=None,
        classes=None,
        class_counts=None,
    ):
        X, y = validate(X, y)

        # TODO: calculate train info for initialization
        n_samples, *shape = X.shape
        order = len(shape)

        # Determine classes  and center
        if classes is None or class_counts is None:
            self.classes_, class_counts = np.unique(y, return_counts=True)
            class_counts = tl.tensor(class_counts)
        else:
            self.classes_ = classes
        if X_centered is None or means is None:
            self.means_, X_centered = center(X, y, self.classes_)
        else:
            self.means_ = means

        # Initialize train info
        self.train_info_ = dict(backward=[])

        # Initialize backward projections and covariances
        # self._init(X,y, classes=classes, class_counts=class_counts, X_centered=X_centered, means=means)
        self._init_backward(X)
        self.scatter_w_ = [None] * order

        # Determine solver parameters
        if self.obj not in OBJECTIVES.keys():
            raise ValueError(f"objective must be one of {list(OBJECTIVES.keys())}")
        solver_params = self.solver_params
        if solver_params is None:
            solver_params = dict()

        # Calculate means and center
        means_centered = self.means_ - tl.mean(self.means_, axis=0)

        # Calculate total scatter
        scatter_t = [None] * order
        for k in range(order):
            scatter_w, shrinkage = mode_scatter(
                X_centered,
                k,
                assume_centered=True,
                shrinkage=self.shrinkage,
                toeplitz=self.toeplitz,
                taper=self.taper,
            )
            scatter_b, _ = mode_scatter(
                self.means_, k, weights=class_counts, shrinkage=0
            )
            scatter_t[k] = scatter_w + scatter_b

        # Iteratively find projections
        shrinkages = [None] * order
        iterator = range(1, self.max_iter + 1)
        if self.verbose:
            iterator = tqdm(iterator, position=0, leave=True)
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

                if not self.refit_shrinkage and shrinkages[k] is not None:
                    shrinkage = shrinkages[k]

                scatter_w, shrinkage = mode_scatter(
                    X_centered_proj,
                    k,
                    assume_centered=True,
                    shrinkage=shrinkage,
                    toeplitz=self.toeplitz,
                    taper=self.taper,
                )
                shrinkages[k] = float(shrinkage)

                # Calculate between class scatter
                means_centered_proj = tl.tenalg.multi_mode_dot(
                    means_centered, self.weights_, modes=modes, skip=k, transpose=True
                )
                scatter_b, _ = mode_scatter(
                    means_centered_proj,
                    k,
                    weights=class_counts,
                    shrinkage=0,
                    assume_centered=True,
                )

                # Solve
                u = self.weights_[k]
                A, B = OBJECTIVES[self.obj](
                    scatter_b,
                    scatter_w,
                    self.weights_[k],
                )

                u, w = solve_gevdh(
                    A,
                    B=B,
                    rank=self.rank_[k],
                    solver=self.solver,
                    which="LA",
                    init=tl.copy(self.weights_[k]),
                    **solver_params,
                )

                # Re-orthogonalize
                u, w = solve_gevdh(
                    u @ u.T @ scatter_t[k] @ u @ u.T,
                    rank=self.rank_[k],
                    solver=self.solver,
                    which="LA",
                    init=tl.copy(self.weights_[k]),
                    **solver_params,
                )

                # Calculate update and check convergence
                if u.shape[-1] != self.weights_[k].shape[-1]:
                    update = np.inf
                else:
                    old = flip_signs(self.weights_[k])
                    new = flip_signs(u)
                    update = tl.norm(new - old)
                    update /= tl.norm(old)

                converged = update < self.tol and converged

                #if np.any(np.isnan(u)):
                #    raise LinAlgError("NaN in weights")

                self.weights_[k] = u

                # Store mode training information
                train_info_row = dict(
                    iteration=self.iter_,
                    mode=k + 1,
                    flip=(self.iter_ - 1) * order + k + 1,
                    update=float(update),
                    shrinkage=float(shrinkage),
                    objective=float(tl.sum(tl.abs(w))),
                )
                self.scatter_w_[k] = scatter_w
                if self.extra_train_info:
                    Xt = self.transform(X)
                    train_info_row.update(backward_stats(Xt, y))
                self.train_info_["backward"].append(train_info_row)

            if self.verbose:
                iterator.set_description(f"Backward HODA model rank={self.rank_}")
            # Exit if converged
            if converged:
                break
        # Convert train_info to list dict
        self.train_info_["backward"] = {
            k: [dic[k] for dic in self.train_info_["backward"]]
            for k in self.train_info_["backward"][0]
        }
        if not converged:
            warnings.warn("Maximum number of iterations reached without convergence")

    def fit_forward(self, X, y, X_centered=None, Xt=None):
        X, y = validate(X, y)

        n_samples, *shape = X.shape
        order = len(shape)

        # Project
        if Xt is None:
            Xt = self.transform(X)

        # Initialize
        self._init_forward()
        self.train_info_["forward"] = []

        # Calculate forward model
        iterator = range(1, self.max_iter)
        if self.verbose:
            iterator = tqdm(iterator, position=0, leave=True)
            iterator.set_description("Forward model ")
        update = np.inf

        shrinkages = [None] * order
        for i in iterator:
            converged = True
            for k in range(order):
                # Partiallu project core tensor
                modes = range(1, order + 1)
                G = tl.tenalg.multi_mode_dot(Xt, self.aps_, modes=modes, skip=k)
                # Regress actvation pattern
                #Xk = tl.unfold(X, k + 1)
                #Gk = tl.unfold(G, k + 1)
                ## TODO: regularization
                #lambda_ = 0.0
                #ap = ridge_regression(Gk.T, Xk.T, lambda_=lambda_).T

                # Least squares regression
                modes = tuple([kk for kk in range(order+1) if kk != k+1])
                XTX = tl.tenalg.tensordot(G,G,modes)
                XTY = tl.tenalg.tensordot(G,X, modes)
                ap =  tl.solve(XTX, XTY).T


                # Calculate update
                update = tl.norm(ap - self.aps_[k])
                update /= tl.norm(self.aps_[k])

                # if k < order-1:
                #    ap *= tl.sign(ap[0,0])
                #if np.any(np.isnan(ap)):
                #    raise LinAlgError("NaN in aps")
                self.aps_[k] = ap
                converged = update < self.tol and converged

                # Store training info
                train_info_row = dict(
                    iteration=i,
                    mode=k + 1,
                    flip=(i - 1) * order + k + 1,
                    update=float(update),
                    #lambda_=float(lambda_),
                )
                if self.extra_train_info:
                    Xt = self.transform(X)
                    X_approx = self.inv_transform(Xt)
                    train_info_row.update(forward_stats(X, Xt, X_approx, y))
                self.train_info_["forward"].append(train_info_row)

            if converged:
                break

        self.train_info_["forward"] = {
            k: [dic[k] for dic in self.train_info_["forward"]]
            for k in self.train_info_["forward"][0]
        }

    def _init_backward(self, X):
        _, *shape = X.shape
        order = X.ndim - 1
        solver_params = self.solver_params
        if solver_params is None:
            solver_params = dict()

        rank = self.rank
        if isinstance(rank, np.integer) or isinstance(rank, int):
            rank = [rank for k in range(order)]
        elif rank is None:
            rank = [None for k in range(order)]

        self.weights_ = [None] * order

        for k in range(order):
            scatter_t, _ = mode_scatter(X, k, assume_centered=True, shrinkage=0)
            u, w = solve_gevdh(
                scatter_t,
                solver=self.solver,
                which="LA",
                **solver_params,
            )
            order = tl.argsort(-w)
            w = w[order]
            u = u[:, order]
            # Calculate explained variance
            explained_var = tl.cumsum(w / tl.sum(w))
            # Determine rank
            if rank[k] is None:
                rank[k] = np.count_nonzero(explained_var < self.theta) + 1
            u = u[:, : rank[k]]
            # Initialize weights
            self.weights_[k] = u

    def _init_forward(self):
        self.aps_ = []
        for w in self.weights_:
            self.aps_.append(tl.copy(w))

    def transform(self, X, y=None):
        X, y = validate(X, y)
        order = len(X.shape) - 1
        Xt = tl.tenalg.multi_mode_dot(
            X, self.weights_, modes=range(1, order + 1), transpose=True
        )
        return Xt

    def inv_transform(self, Xt, y=None):
        X, y = validate(Xt, y)
        order = Xt.ndim - 1
        modes = [k + 1 for k in range(order)]
        return tl.tenalg.multi_mode_dot(Xt, self.aps_, modes=modes)

    @property
    def rank_(self):
        order = len(self.weights_)
        return tuple([self.weights_[k].shape[-1] for k in range(order)])

    @property
    def n_params_(self):
        return sum([s.size for s in self.weights_])


def backward_stats(Xt, y):
    n, *shape = Xt.shape
    p = math.prod(shape)
    stats = dict()
    # stats["F_tr"] = float(f_multiway(Xt, y, method="tr"))
    # stats["F_rt"] = float(f_multiway(Xt, y, method="rt"))

    # lda = LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")
    # Xtf = tl.to_numpy(Xt.reshape((len(Xt), -1)))
    # lda.fit(Xtf, y)
    # y_pred = lda.predict_proba(Xtf)
    # stats["log_loss"] = float(log_loss(y, y_pred))
    # _, y_num = np.unique(y, return_inverse=True)
    # logit = Logit(y_num,tl.to_numpy(Xt.reshape(n,p)))
    # logit_res = logit.fit()
    # stats["log_like"] = logit_res.llf
    # pR2 = logit_res.prsquared
    # stats["pseudo_R2"] = pR2
    # pR2_adj = 1-(1-pR2)*(n-1)/(n-p-1)
    # stats["pseudo_R2_adj"] = pR2_adj
    # stats["aic"] = logit_res.aic
    # stats["bic"] = logit_res.bic
    return stats


def forward_stats(X, Xt, X_approx, y):
    # err = X - X_approx
    # n_samples = X.shape[0]
    # G_flat = tl.to_numpy(
    #    Xt.reshape((n_samples, -1), order="F"),
    # )
    # err_flat = tl.to_numpy(err.reshape((n_samples, -1), order="F"))
    # cross_corr = np.corrcoef(G_flat, err_flat, rowvar=False)
    # cross_corr = cross_corr[G_flat.shape[-1] :, : G_flat.shape[-1]].T
    # cross_corr = tl.metrics.regression.MSE(cross_corr, 0)
    mse = tl.metrics.regression.MSE(X, X_approx)
    nmse = mse / tl.metrics.regression.MSE(X, 0)
    stats = dict()
    stats["mse"] = float(mse)
    stats["nmse"] = float(nmse)
    # stats["err_cross_corr"] = float(cross_corr)
    return stats


class BTTDA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        ranks=None,
        hoda_params=None,
        extra_train_info=False,
        verbose=False,
        forward=True,
    ):
        self.hoda_params = hoda_params
        self.verbose = verbose
        self.extra_train_info = extra_train_info
        self.forward = forward
        self.ranks = ranks

    def fit(self, X, y=None, blocks=None):
        X, y = validate(X, y)
        n_samples, *shape = X.shape

        self.classes_, class_counts = np.unique(y, return_counts=True)
        class_counts = tl.tensor(class_counts)

        hoda_params = self.hoda_params
        if hoda_params is None:
            hoda_params = dict()
        self.blocks_ = []
        self.train_info_ = []

        err = tl.copy(X)
        for b, rank in enumerate(self.ranks):
            try:
                if blocks is not None and b < len(blocks):
                    # TODO error if ranks are not equal
                    block = blocks[b]
                else:
                    if self.verbose:
                        print(f"Fitting block {b+1}/{len(self.ranks)}...")
                    hoda_params["rank"] = rank
                    block = HODA(**hoda_params)
                    block.fit_backward(
                        err,
                        y,
                        classes=self.classes_,
                        class_counts=class_counts,
                    )

                self.blocks_.append(block)
                G = block.transform(err)
                block.fit_forward(err, y, Xt=G)
                err -= block.inv_transform(G)

                # Calculate train info
                train_info_row = dict()
                train_info_row["block"] = self.n_blocks_
                train_info_row["rank"] = block.rank_

                if self.extra_train_info:
                    Xt = self.transform(X)
                    X_approx = self.inv_transform(Xt)
                    train_info_row.update(backward_stats(Xt, y))
                    train_info_row.update(forward_stats(X, Xt, X_approx, y))
                self.train_info_.append(train_info_row)
            except (LinAlgError, ValueError) as e:
                warnings.warn(RuntimeWarning(str(e)))
                break

        # Convert train info to list dict
        # self.train_info_ = {
        #    k: [dic[k] for dic in self.train_info_] for k in self.train_info_[0]
        # }
        return self

    @property
    def n_blocks_(self):
        return len(self.blocks_)

    @property
    def n_params_(self):
        return sum([b.n_params_ for b in self.blocks_])

    def transform(
        self, X, y=None, blocks=None, n_blocks=None, return_err=False, flatten=True, **_
    ):
        X, y = validate(X, y)
        n_samples, *_ = X.shape

        if blocks is None:
            blocks = self.blocks_
        if n_blocks is not None:
            blocks = blocks[:n_blocks]

        Gs = []

        err = tl.copy(X)
        for block in blocks:
            G = block.transform(err)
            Gs.append(G)
            err -= block.inv_transform(G)

        # res = tl.copy(X)
        # for b, block in enumerate(blocks):
        #   new_res = res - block.inv_transform(block.transform(res))
        #   diff = res-new_res
        #   res = new_res
        #   G = block.transform(diff)
        #   Gs.append(G)

        if flatten:
            Gs = [G.reshape((n_samples, -1)) for G in Gs]
            Gs = tl.concatenate(Gs, axis=1)
        if return_err:
            return Gs, err
        return Gs

    def inv_transform(self, Xt, y=None, n_blocks=None):
        X, y = validate(Xt, y)
        n_samples, _ = Xt.shape
        if n_blocks is None:
            n_blocks = self.n_blocks_
        n_blocks = min(n_blocks, self.n_blocks_)
        shape = (n_samples, *[s.shape[0] for s in self.blocks_[0].weights_])

        X = tl.zeros(shape)
        for b in range(n_blocks):
            block = self.blocks_[b]
            n_features = math.prod(block.rank_)
            Xtb = Xt[:, :n_features]
            Xtb = Xtb.reshape((n_samples, *block.rank_))
            X += block.inv_transform(Xtb)
            Xt = Xt[:, n_features:]
        return X

    @property
    def ranks_(self):
        return tuple([b.rank_ for b in self.blocks_])


def f_multiway(
    X,
    y=None,
    classes=None,
    class_counts=None,
    assume_centered=False,
    means=None,
    method="tr",
):
    n_samples, *shape = X.shape
    n_features = math.prod(shape)
    if not tl.is_tensor(X):
        X = tl.tensor(X)
    if classes is None or class_counts is None:
        classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    # Calculate class means, overall class mean and center data
    if assume_centered:
        X_centered = X
        if means is None:
            raise ValueError("must specify means when assume_centered=True")
    else:
        means, X_centered = center(X, y, classes)

    class_mean = tl.mean(means, axis=0)
    if method == "tr":
        tr_scatter_w = norm_fro(X_centered) ** 2
        tr_scatter_b = 0
        for ci, c in enumerate(classes):
            mean_centered = means[ci] - class_mean
            tr_scatter_b += class_counts[ci] * norm_fro(mean_centered) ** 2
        # Calculate Fisher ratio
        F = tr_scatter_b / tr_scatter_w  # * ((n_classes - 1) / (n_samples - n_classes))
    elif method == "rt":
        X_centered_flat = tl.unfold(X_centered, 0)

        scatter_w, _ = mode_scatter(
            X_centered_flat, 0, assume_centered=True, shrinkage="lw"
        )
        means_flat = tl.unfold(means, 0)
        scatter_b, _ = mode_scatter(means_flat, 0, assume_centered=False, shrinkage=0)
        w = solve_gevdh(
            scatter_b,
            scatter_w,
            rank=X_centered_flat.shape[-1],
            solver="lanczos",
            eigvals_only=True,
        )
        F = tl.sum(w)
    else:
        raise ValueError(
            "method must be either 'tr' (trace-ratio) or 'rt' (ratio-trace)"
        )
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
    F = msb / msw
    p = fdtrc(dfbn, dfwn, F)
    return F, p
