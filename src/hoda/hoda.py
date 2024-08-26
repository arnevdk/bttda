import itertools
from sklearn.base import clone
import math
import pdb

import numpy as np
import pandas as pd
import tensorly as tl
import tensorly.decomposition
import tensorly.tenalg
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import (ElasticNet, LogisticRegression,
                                  LogisticRegressionCV, Ridge)
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
# from statsmodels.discrete.discrete_model import Logit
#from tqdm.notebook import tqdm

from hoda.backend import copy, lstsq, pinv
from hoda.classification import SelectF
from hoda.cov import KroneckerCovariance, mode_scatter
from hoda.tensorize import Vectorize, vec
from hoda.util import center, f_multiway, lstsq_ridge, r_squared, trunc_eigh

from tqdm import tqdm


def obj_rt(scatter_b, scatter_w, _):
    """Ratio trace objective Tr(uT Sw^-1 Sb u).

    Phan, A. H., & Cichocki, A. (2010).
    Tensor decompositions for feature extraction and classification of high
    dimensional datasets. Nonlinear theory and its applications, IEICE, 1(1), 37-68.

    Wang, H., Yan, S., Xu, D., Tang, X., & Huang, T. (2007, June). Trace ratio
    vs. ratio trace for dimensionality reduction. In 2007 IEEE Conference on
    Computer Vision and Pattern Recognition (pp. 1-8). IEEE.

    """
    return scatter_b, scatter_w, True


def obj_tr(scatter_b, scatter_w, u, psi=1):
    """Trace ratio objective Tr(uT Sb u)/Tr(uT Sw u).

    Phan, A. H., & Cichocki, A. (2010).
    Tensor decompositions for feature extraction and classification of high
    dimensional datasets. Nonlinear theory and its applications, IEICE, 1(1), 37-68.

    Wang, H., Yan, S., Xu, D., Tang, X., & Huang, T. (2007, June). Trace ratio
    vs. ratio trace for dimensionality reduction. In 2007 IEEE Conference on
    Computer Vision and Pattern Recognition (pp. 1-8). IEEE.
    """
    phi = tl.trace(u.T @ scatter_b @ u) / tl.trace(u.T @ scatter_w @ u)
    # _, phi = trunc_eigh(scatter_b, scatter_w, rank=1, largest=True)
    A = scatter_b - psi * phi * scatter_w
    return A, None, True


def obj_lfl(scatter_b, scatter_w, u, psi=1):
    """Linear feature learning obbjective.

    Aghili, S. N., Kilani, S., Khushaba, R. N., & Rouhani, E. (2023).
    A spatial-temporal linear feature learning algorithm for P300-based -
    brain-computer interfaces. Heliyon, 9(4).
    """
    phi = tl.trace(u.T @ scatter_b @ u) / tl.trace(u.T @ scatter_w @ u)
    A = scatter_b - phi * scatter_w
    B = scatter_w
    return A, B, True


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


class HODA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(
        self,
        max_iter=256,
        tol=1e-8,
        rank=None,
        init="eye",
        shrinkage="lw",
        toeplitz=None,
        taper=False,
        obj="rt",
        solver="lanczos",
        verbose=False,
        solver_params=None,
        extra_train_info=False,
        random_state=None,
        delta=None,
        forward=False,
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
        self.delta = delta
        self.forward = forward

    def fit(self, X, y, classes=None, class_counts=None):
        assert tl.is_tensor(X)
        # Convert to tensor
        # if not tl.is_tensor(X):
        #    X = tl.tensor(X)
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
        self, X, y, X_centered=None, means=None, classes=None, class_counts=None
    ):
        assert tl.is_tensor(X)
        # TODO: calculate train info for initialization
        # Convert to tensor
        # if not tl.is_tensor(X):
        #    X = tl.tensor(X)
        n_samples, *shape = X.shape
        order = len(shape)

        # Determine classses  and center
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
        self._init(X)
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
            scatter_w, _ = mode_scatter(
                X_centered,
                k,
                assume_centered=True,
                shrinkage=self.shrinkage,
                toeplitz=self.toeplitz,
                taper=self.taper,
            )
            scatter_b, _ = mode_scatter(
                self.means_, k, weights=tl.sqrt(class_counts), shrinkage=0
            )
            scatter_t[k] = scatter_w + scatter_b

        # Iteratively find projections
        iterator = range(1, self.max_iter + 1)
        if self.verbose:
            iterator = tqdm(iterator)
            iterator.set_description(f"Backward HODA model rank={self.rank_}")
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

                # Calculate between class scatter
                means_centered_proj = tl.tenalg.multi_mode_dot(
                    means_centered, self.weights_, modes=modes, skip=k, transpose=True
                )
                scatter_b, _ = mode_scatter(
                    means_centered_proj,
                    k,
                    weights=tl.sqrt(class_counts),
                    shrinkage=0,
                    assume_centered=True,
                )

                # Solve
                A, B, largest = OBJECTIVES[self.obj](
                    scatter_b, scatter_w, self.weights_[k]
                )
                if self.solver == "lobpcg":
                    solver_params["init"] = copy(self.weights_[k])
                u, w = trunc_eigh(
                    A,
                    B=B,
                    rank=self.rank_[k],
                    method=self.solver,
                    largest=largest,
                    solver_params=solver_params,
                )
                if self.delta is not None:
                    idc = np.argsort(w)
                    w = w[idc]
                    u = u[:, idc]
                    u = u[:, tl.cumsum(w) / tl.sum(w) > self.delta]
                new_rank = u.shape[-1]
                # Re-orthogonalize
                if self.solver == "lobpcg":
                    solver_params["init"] = u
                u, w = trunc_eigh(
                    u @ u.T @ (scatter_t[k]) @ u @ u.T,
                    # rank=self.rank_[k],
                    rank=new_rank,
                    method=self.solver,
                    largest=largest,
                    solver_params=solver_params,
                )
                # Flip sign
                sign = tl.sign(u[0, :])
                u *= sign
                # Normalize
                u = u / tl.norm(u, axis=0)

                # Calculate update and check convergence
                if u.shape[-1] != self.weights_[k].shape[-1]:
                    update = np.inf
                else:
                    update = tl.norm(u - self.weights_[k])
                    update /= tl.norm(self.weights_[k])

                converged = update < self.tol and converged

                # Update weights
                self.weights_[k] = u

                # Store mode training information
                train_info_row = dict(
                    iteration=self.iter_,
                    mode=k,
                    update=float(update),
                    shrinkage=float(shrinkage),
                    rank=new_rank,
                    objective=float(tl.sum(w)),
                )
                self.scatter_w_[k] = scatter_w
                if self.extra_train_info:
                    Xt = self.transform(X)
                    train_info_row.update(backward_stats(Xt, y))
                self.train_info_["backward"].append(train_info_row)

            # Exit if converged
            if converged:
                break
        # Convert train_info to list dict
        self.train_info_["backward"] = {
            k: [dic[k] for dic in self.train_info_["backward"]]
            for k in self.train_info_["backward"][0]
        }

    def fit_forward(self, X, y, X_centered=None, Xt=None, Xt_centered=None):
        assert tl.is_tensor(X)
        n_samples, *shape = X.shape
        order = len(shape)

        # Project

        if Xt is None:
            Xt = self.transform(X)
        # if Xt_centered is None:
        #    _, Xt_centered = center(Xt, y)
        # if X_centered is None:
        #    _, X_centered = center(X, y)

        # Apply class balance weights
        #classes, class_counts = np.unique(y, return_counts=True)
        #weights = tl.zeros(n_samples)
        #for c, cls in enumerate(classes):
        #    weights[y == cls] = 1 / class_counts[c]
        #weights = tl.sqrt(weights)
        #modes = tuple([k + 1 for k in range(order)])
        #weights = np.expand_dims(weights, axis=modes)
        #X = weights * X
        #Xt = weights * Xt

        # Initialize
        self.aps_ = [copy(w) for w in self.weights_]
        self.train_info_["forward"] = []

        # Calculate forward model
        iterator = range(0, self.max_iter)
        if self.verbose:
            iterator = tqdm(iterator)
            iterator.set_description("Forward model ")
        update = np.inf
        for i in iterator:
            converged = True
            for k in range(order):
                # Initialize or calculate activation pattern
                if not i:
                    ap = copy(self.weights_[k])
                    # ap = tl.eye(shape[k])[:, : self.rank_[k]]
                    update = np.inf
                    shrink = np.nan
                else:
                    modes = range(1, order + 1)
                    G = tl.tenalg.multi_mode_dot(Xt, self.aps_, modes=modes, skip=k)

                    #modes = [0] + [kk + 1 for kk in range(order) if kk != k]
                    #cov_cross = tl.tensordot(X, G, axes=(modes, modes))
                    #pdb.set_trace()
                    #cov_g, shrink = mode_scatter(
                    #        G, k,
                    #        shrinkage=self.shrinkage,
                    #        toeplitz=self.toeplitz,
                    #        assume_centered=True,
                    #)
                    #ap = tl.solve(cov_g.T, cov_cross.T).T


                    Gk = tl.unfold(G, k + 1)
                    Xk = tl.unfold(X, k + 1)
                    ap, residuals, rank, s = lstsq(Gk.T, Xk.T)
                    #ap = lstsq_ridge(Gk.T, Xk.T, lambda_=0)
                    ap = ap.T
                    shrink = 0

                    update = tl.norm(ap - self.aps_[k])
                    update /= tl.norm(self.aps_[k])
                self.aps_[k] = ap
                converged = update < self.tol and converged

                # Store training info
                train_info_row = dict()
                train_info_row["iteration"] = i
                train_info_row["mode"] = k
                train_info_row["update"] = float(update)
                train_info_row["shrinkage"] = float(shrink)
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

    def _init(self, X):
        _, *shape = X.shape
        order = X.ndim - 1

        # Determine multilinear rank
        modes = [k + 1 for k in range(order)]
        rank = self.rank
        if isinstance(rank, np.integer) or isinstance(rank, int):
            rank = [self.rank for k in range(order)]
        elif self.rank is None:
            rank = shape
        else:
            rank = self.rank

        for k in range(order):
            if rank[k] > shape[k]:
                raise ValueError(
                    f"mode rank must be less or equal to dimension ({rank[k]} > {shape[k]})"
                )

        if self.init == "mlsvd":
            (_, self.weights_), _ = tl.decomposition.partial_tucker(
                X,
                rank=rank,
                modes=modes,
            )
        else:
            self.weights_ = [None] * order
            for k in range(order):
                if self.init == "svd":
                    Xk = tl.unfold(X, k + 1)
                    self.weights_[k], _, _ = tl.tenalg.svd_interface(
                        Xk, method="truncated_svd", n_eigenvecs=rank[k]
                    )
                elif self.init == "random":
                    self.weights_[k] = tl.random.base.random_tensor(
                        (shape[k], rank[k]), random_state=self.random_state
                    )
                    self.weights_[k], _ = tl.qr(self.weights_[k], mode="reduced")
                elif self.init == "eye":
                    self.weights_[k] = tl.eye((shape[k]))[:, : rank[k]]
                elif isinstance(self.init, list):
                    if tl.is_tensor(self.init[k]):
                        self.weights_[k] = self.init[k]
                    else:
                        self.weights_[k] = tl.tensor(self.init[k])
                else:
                    raise ValueError(
                        "init must be one of ['mlsvd', 'svd', 'random', 'eye'] or a list of weight matrices"
                    )
        # for k in range(order):
        #    self.weights_[k] = self.weights_[k] / tl.norm(self.weights_[k])

    def transform(self, X, y=None):
        # if not tl.is_tensor(X):
        #    X = tl.tensor(X)
        order = len(X.shape) - 1
        Xt = tl.tenalg.multi_mode_dot(
            X, self.weights_, modes=range(1, order + 1), transpose=True
        )
        return Xt

    def inv_transform(self, Xt, y=None):
        # if not tl.is_tensor(Xt):
        #    Xt = tl.tensor(Xt)
        order = Xt.ndim - 1
        modes = [k + 1 for k in range(order)]
        return tl.tenalg.multi_mode_dot(Xt, self.aps_, modes=modes)
        # return tl.tenalg.tensordot(Xt, self.A_, modes=[(1, 2), (2, 3)])

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
    stats["F_tr"] = float(f_multiway(Xt, y, method="tr"))
    stats["F_rt"] = float(f_multiway(Xt, y, method="rt"))
    _, y_num = np.unique(y, return_inverse=True)
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
        self.ranks = ranks
        self.verbose = verbose
        self.extra_train_info = extra_train_info
        self.forward=forward

    def fit(self, X, y=None, blocks=None):
        assert tl.is_tensor(X)
        n_samples, *shape = X.shape

        self.classes_, class_counts = np.unique(y, return_counts=True)
        class_counts = tl.tensor(class_counts)

        hoda_params = self.hoda_params
        if hoda_params is None:
            hoda_params = dict()
        self.blocks_ = []
        self.train_info_ = []

        err = copy(X)
        for b, rank in enumerate(self.ranks):
            if blocks is not None and b < len(blocks):
                # TODO error if ranks are not equal
                block = blocks[b]
            else:
                if self.verbose:
                    print(f"Fitting block {b+1}/{len(self.ranks)}...")
                hoda_params["rank"] = rank
                block = HODA(**hoda_params)
                block.fit_backward(
                    err, y, classes=self.classes_, class_counts=class_counts
                )
            self.blocks_.append(block)
            G = block.transform(err)
            if b < len(self.ranks)-1 or self.forward:
                if not hasattr(block,'aps_'):
                    block.fit_forward(err, y, Xt=G)
                err -= block.inv_transform(G)
            # Calculate train info
            train_info_row = dict()
            train_info_row["block"] = self.n_blocks_
            train_info_row["rank"] = block.rank_
            if self.extra_train_info:
                Xt = self.transform(X)
                if b < len(self.ranks)-1 or self.forward:
                    X_approx = self.inv_transform(Xt)
                else:
                    X_approx = np.zeros_like(X)
                train_info_row.update(backward_stats(Xt, y))
                train_info_row.update(forward_stats(X, Xt, X_approx, y))
            self.train_info_.append(train_info_row)

        # Convert train info to list dict
        self.train_info_ = {
            k: [dic[k] for dic in self.train_info_] for k in self.train_info_[0]
        }
        return self

    @property
    def n_blocks_(self):
        return len(self.blocks_)

    @property
    def n_params_(self):
        return sum([b.n_params_ for b in self.blocks_])

    def transform(self, X, y=None, blocks=None, return_err=False,**_):
        assert tl.is_tensor(X)
        err = copy(X)
        n_samples, *_ = X.shape
        Xt = []
        if blocks is None:
            blocks = self.blocks_
        for b, block in enumerate(blocks):
            Xtb = block.transform(err, y)
            Xt.append(Xtb.reshape(n_samples, -1))
            if b < len(blocks)-1:
                err -= block.inv_transform(Xtb)

        Xt = tl.concatenate(Xt, axis=1)
        if return_err:
            return Xt, err
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
            n_features = math.prod(block.rank_)
            Xtb = Xt[:, :n_features]
            Xtb = Xtb.reshape((n_samples, *block.rank_))
            X += block.inv_transform(Xtb)
            Xt = Xt[:, n_features:]
        return X

    @property
    def ranks_(self):
        return tuple([b.rank_ for b in self.blocks_])


class GreedyBTTDA(BTTDA):
    def __init__(
        self,
        hoda_params=None,
        verbose=False,
        extra_train_info=False,
        max_blocks=16,
        cv=None,
        rank_grid=None,
        truncate=True,
        n_jobs=None,
        clf = None,
    ):
        self.hoda_params = hoda_params
        self.verbose = verbose
        self.extra_train_info = extra_train_info
        self.max_blocks = max_blocks
        self.cv = cv
        self.rank_grid = rank_grid
        self.truncate = truncate
        self.n_jobs = n_jobs
        self.clf=clf
        super().__init__(
            hoda_params=hoda_params, verbose=verbose, extra_train_info=extra_train_info
        )

    def log_rank_grid(self, shape):
        order = len(shape)
        grid = []
        max_r = int(np.floor(np.log2(max(shape)) + 1))
        for r in range(max_r):
            rank = [2**r] * order
            for k in range(order):
                rank[k] = min(rank[k], shape[k])
            grid.append(tuple(rank))
        #grid.append(tuple(shape))
        grid = sorted(list(set(grid)))
        return grid

    def fit(self, X, y, test=False):
        assert tl.is_tensor(X)
        n_samples, *shape = X.shape

        # Initialize
        cv = self.cv
        if cv is None:
            cv = StratifiedKFold(shuffle=True)
        rank_grid = self.rank_grid
        if rank_grid is not None:
            rank_grid = self.rank_grid or self.log_rank_grid(shape)
        self.model_select_info_ = []

        # Create cross validation splits
        all_idc = np.arange(len(X))
        if test:
            idc, test_idc, _, _ = train_test_split(all_idc, y, test_size=0.2)
            splits = list(cv.split(idc, y[idc]))
            for f, (train_idc, val_idc) in enumerate(splits):
                splits[f] = (idc[train_idc], idc[val_idc], test_idc)
        else:
            splits = list(cv.split(X, y))
            for f, (train_idc, val_idc) in enumerate(splits):
                splits[f] = (train_idc, val_idc, None)

        
        # Instantiate cross validation models
        fold_bttda = []
        hoda_params = self.hoda_params
        hoda_params['verbose'] = False
        for _ in splits:
            fold_bttda.append(BTTDA(
                ranks=None,
                hoda_params=self.hoda_params,
                extra_train_info=False,
                verbose=False,
                forward=False,
        ))

        # Greedy hyperparameter search
        ranks = []
        for b in range(self.max_blocks):
            if self.verbose:
                print(f"Model selection block {b+1}/{self.max_blocks}...")
                print()

            # Evaluate different ranks
            rank_grid = self.log_rank_grid(shape)
            val_scores = np.zeros((len(splits), len(rank_grid)))
            for ri,r in enumerate(rank_grid):
                if self.verbose:
                    print(f"Trying rank {r}", end='\t')

                ranks_candidate = ranks + [r]
                fold_Xt = []
                for fold, (train_idc, val_idc, test_idc) in enumerate(splits):
                    fold_bttda[fold].set_params(forward=False)
                    fold_bttda[fold].set_params(ranks=ranks_candidate)
                    blocks = fold_bttda[fold].blocks_[:b] if b else []
                    fold_bttda[fold] = fold_bttda[fold].fit(
                        X[train_idc],y[train_idc], blocks=blocks
                    )
                    fold_Xt.append(fold_bttda[fold].transform(X))
                
                for fold, (train_idc, val_idc, test_idc) in enumerate(splits):
                    Xt = fold_Xt[fold]
                    clf = clone(self.clf).fit(Xt[train_idc], y[train_idc]) 
                    y_pred = clf.decision_function(Xt)
                    train_score = roc_auc_score(y[train_idc], y_pred[train_idc])
                    val_score = roc_auc_score(y[val_idc], y_pred[val_idc])
                    val_scores[fold, ri] = val_score
                    res = dict(
                        block=b,
                        rank=r,
                        fold=fold,
                        train_score=train_score,
                        val_score=val_score,
                        n_features = Xt.shape[-1]
                    )
                    if test:
                        res['test_score'] = roc_auc_score(y[test_idc], y_pred[test_idc])
                    self.model_select_info_.append(res)

                if self.verbose:
                    print(f'score: {np.mean(val_scores[:,ri]):.4f}')

            best_rank = rank_grid[np.argmax(np.mean(val_scores, axis=0))]
            best_score = np.max(np.mean(val_scores, axis=0))
            ranks.append(best_rank)
            print()
            print(f"Selected rank {best_rank} with score {best_score:.4f}")
            print(f"New ranks: {ranks}")
            print()
            print()
            
            # Fit forward
            for fold, (train_idc, val_idc, test_idc) in enumerate(splits):
                fold_bttda[fold].set_params(ranks=ranks)
                fold_bttda[fold].set_params(forward=True)
                blocks=fold_bttda[fold].blocks_[:b] if b else []
                fold_bttda[fold] = fold_bttda[fold].fit(
                        X[train_idc],y[train_idc], blocks=blocks,
                )

                    
        self.ranks = ranks
        self.model_select_info_ = pd.DataFrame(self.model_select_info_)
        self.model_select_info_ = self.model_select_info_.set_index(["block", "rank"])

        # Truncate and determine best features
        if self.truncate:
            df = self.model_select_info_best_
            df = df.groupby(["block", "rank"])
            df = df["val_score"].aggregate("mean")
            best_block, _ = df.idxmax()
            best_n_blocks = best_block + 1
            ranks = ranks[:best_n_blocks]

        # Train BTTDA
        self.ranks = ranks
        super().fit(X, y)
        # Train select
        Xt = self.transform(X, select=False)
        # _, self.weights, self.orth  = tl.tenalg.svd_interface(Xt[train_idc])
        # Xt = Xt@self.orth.T@tl.diag(1/self.weights)
        return self

    @property
    def model_select_info_best_(self):
        df_agg = self.model_select_info_.groupby(["block", "rank"])
        df_agg = df_agg["val_score"].aggregate("mean")
        idc = df_agg.groupby("block").idxmax()
        df_select = self.model_select_info_.loc[idc]
        return df_select
