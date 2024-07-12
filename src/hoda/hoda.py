import itertools
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
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm

from hoda.backend import copy, lstsq, pinv
from hoda.classification import SelectF
from hoda.cov import KroneckerCovariance, mode_scatter
from hoda.tensorize import Vectorize, vec
from hoda.util import center, f_multiway, lstsq_ridge, trunc_eigh


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
                    train_info_row.update(
                        self._extra_train_info_backward(
                            X,
                            y,
                            class_counts,
                        )
                    )
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
        if Xt_centered is None:
            _, Xt_centered = center(Xt, y)
        if X_centered is None:
            _, X_centered = center(X, y)

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
                    update = np.inf
                    shrink = np.nan
                else:
                    modes = range(1, order + 1)
                    G = tl.tenalg.multi_mode_dot(
                        Xt_centered, self.aps_, modes=modes, skip=k
                    )

                    # modes = [0] + [kk + 1 for kk in range(order) if kk != k]
                    # cov_cross = tl.tensordot(X_centered, G, axes=(modes, modes))
                    # cov_g, shrink = mode_scatter(
                    #    G,
                    #    k,
                    #    shrinkage=self.shrinkage,
                    #    # shrinkage=0,
                    #    assume_centered=True,
                    # )
                    # ap = tl.solve(cov_g.T, cov_cross.T).T

                    Gk = tl.unfold(G, k + 1)
                    Xk = tl.unfold(X_centered, k + 1)
                    ap, *_ = lstsq(Gk.T, Xk.T)
                    # ap = lstsq_ridge(Gk.T, Xk.T, lambda_=0)
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
                    train_info_row.update(self._extra_train_info_forward(X, Xt, y))
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

    def _extra_train_info_backward(self, X, y, class_counts):
        Xt = self.transform(X)
        info = dict()
        # Objective: multi-way F-score
        # trace-ratio
        F_tr = f_multiway(
            Xt,
            y,
            self.classes_,
            class_counts,
            method="tr",
        )
        info["F_tr"] = float(F_tr)
        # ratio-trace
        F_rt = f_multiway(
            Xt,
            y,
            self.classes_,
            class_counts,
            method="rt",
        )
        info["F_rt"] = float(F_rt)
        return info

    def _extra_train_info_forward(self, X, Xt, y):
        X_approx = self.inv_transform(Xt)
        err = X - X_approx

        n_samples = X.shape[0]
        G_flat = tl.to_numpy(
            Xt.reshape((n_samples, -1), order="F"),
        )
        err_flat = tl.to_numpy(err.reshape((n_samples, -1), order="F"))
        cross_corr = np.corrcoef(G_flat, err_flat, rowvar=False)
        cross_corr = cross_corr[G_flat.shape[-1] :, : G_flat.shape[-1]].T
        cross_corr = tl.metrics.regression.MSE(cross_corr, 0)
        mse = tl.metrics.regression.MSE(X, X_approx)
        nmse = mse / tl.metrics.regression.MSE(X, 0)
        info = dict()
        info["mse"] = float(mse)
        info["nmse"] = float(nmse)
        info["err_cross_corr"] = float(cross_corr)
        return info


class BTTDA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        ranks=None,
        hoda_params=None,
        extra_train_info=False,
        verbose=False,
    ):
        self.hoda_params = hoda_params
        self.ranks = ranks
        self.verbose = verbose
        self.extra_train_info = extra_train_info

    def fit(self, X, y=None, blocks=None, X_test=None, y_test=None):
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
            block.fit_forward(err, y, Xt=G)
            err -= block.inv_transform(G)
            # Calculate train info
            train_info_row = dict()
            train_info_row["block"] = self.n_blocks_
            train_info_row["rank"] = block.rank_
            if self.extra_train_info:
                train_info_row.update(self._extra_train_info(X, y, class_counts))
            if X_test is not None:
                train_info_row["split"] = "train"
                test_info_row = dict(train_info_row)
                test_info_row.update(
                    self._extra_train_info(X_test, y_test, class_counts)
                )
                test_info_row["split"] = "test"
                self.train_info_.append(test_info_row)
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

    def transform(self, X, y=None, blocks=None, return_err=False):
        assert tl.is_tensor(X)
        err = copy(X)
        n_samples, *_ = X.shape
        Xt = []
        if blocks is None:
            blocks = self.blocks_
        for b, block in enumerate(blocks):
            Xtb = block.transform(err, y)
            Xt.append(Xtb.reshape(n_samples, -1))
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

    def _extra_train_info(self, X, y, class_counts):
        Xt = self.transform(X, select=False)
        X_rec = self.inv_transform(Xt)
        mse = tl.metrics.regression.MSE(X_rec, X)
        nmse = mse / tl.metrics.regression.MSE(X, 0)
        # Objective: multi-way F-score
        # trace-ratio
        F_tr = f_multiway(
            Xt,
            y,
            self.classes_,
            class_counts,
            method="tr",
        )
        # ratio-trace
        F_rt = f_multiway(
            Xt,
            y,
            self.classes_,
            class_counts,
            method="rt",
        )

        info = dict()
        info["mse"] = float(mse)
        info["nmse"] = float(nmse)
        info["F_tr"] = float(F_tr)
        info["F_rt"] = float(F_rt)
        return info


# class GreedyBTTDA(BTTDA):
#    def __init__(
#        self,
#        hoda_params=None,
#        verbose=False,
#        extra_train_info=False,
#        max_blocks=16,
#        cv=None,
#        rank_grid=None,
#        truncate=True,
#        n_jobs=None,
#    ):
#        super().__init__(
#            hoda_params=hoda_params, verbose=verbose, extra_train_info=extra_train_info
#        )
#        self.max_blocks = max_blocks
#        self.cv = cv
#        self.rank_grid = rank_grid
#        self.truncate = truncate
#        self.n_jobs = n_jobs
#
#    def fit(self, X, y=None):
#        assert tl.is_tensor(X)
#        _, *shape = X.shape
#        # Set some defaults
#        cv = self.cv
#        if cv is None:
#            cv = StratifiedKFold(shuffle=True)
#        rank_grid = self.rank_grid
#        if rank_grid is None:
#            rank_grid = [
#                int(r)
#                for r in 2 ** np.arange(np.floor(np.log2(min(shape)) + 1), dtype=int)
#            ]
#
#        # Greedy rank selection
#        self.ranks = []
#        self.val_scores_ = []
#        self.model_select_info_ = []
#        for b in range(self.max_blocks):
#            if self.verbose:
#                print(f"Model selection block {b+1}/{self.max_blocks}...")
#
#            param_grid = dict(bttda__ranks=[self.ranks + [r] for r in rank_grid])
#            clf = self.clf_pipe()
#            gs = GridSearchCV(
#                clf,
#                param_grid,
#                scoring="roc_auc",
#                n_jobs=self.n_jobs,
#                refit=False,
#                verbose=self.verbose,
#                cv=cv,
#            )
#            gs.fit(X, y)
#            info = pd.DataFrame(gs.cv_results_)
#            info["block"] = b
#            self.model_select_info_.append(info)
#            self.ranks = gs.best_params_["bttda__ranks"]
#            if self.verbose:
#                print(f"Selecting ranks {self.ranks}")
#            self.val_scores_.append(gs.best_score_)
#
#        self.model_select_info_ = pd.concat(self.model_select_info_)
#        if self.truncate:
#            best_n_blocks = np.argmax(self.val_scores_) + 1
#            self.ranks = self.ranks[:best_n_blocks]
#        super().fit(X, y)
#        return self
#
#    def clf_pipe(self):
#        bttda_params = dict(
#            hoda_params=self.hoda_params,
#            extra_train_info=False,
#            verbose=self.verbose,
#        )
#        pipe = Pipeline(
#            [
#                ("bttda", BTTDA(**bttda_params)),
#                ("clf", self.clf()),
#            ]
#        )
#        return pipe
#
#    @staticmethod
#    def clf():
#        return make_pipeline(
#            Vectorize(),
#            StandardScaler(),
#            LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr"),
#        )


#class GreedyBTTDA(BTTDA):
#    def __init__(
#        self,
#        hoda_params=None,
#        verbose=False,
#        extra_train_info=False,
#        max_blocks=16,
#        cv=None,
#        rank_grid=None,
#        truncate=True,
#        n_jobs=None,
#        select=True
#    ):
#        self.hoda_params = hoda_params
#        self.verbose = verbose
#        self.extra_train_info = extra_train_info
#
#        self.max_blocks = max_blocks
#        self.cv = cv
#        self.rank_grid = rank_grid
#        self.truncate = truncate
#        self.n_jobs = n_jobs
#        self.select=select
#        super().__init__(
#            hoda_params=hoda_params, verbose=verbose, extra_train_info=extra_train_info
#        )
#
#    def fit(self, X, y, test=False):
#        assert tl.is_tensor(X)
#        n_samples, *shape = X.shape
#
#        cv = self.cv
#        if cv is None:
#            cv = StratifiedKFold(shuffle=True)
#        rank_grid = self.rank_grid
#        if rank_grid is None:
#            rank_grid = list(
#                2 ** np.arange(np.floor(np.log2(min(shape)) + 1), dtype=int)
#            )
#            rank_grid.append(min(shape))
#            rank_grid = sorted(list(set(rank_grid)))
#
#        fold_blocks = []
#        fold_err = []
#        self.model_select_info_ = []
#        # Evaluate folds and ranks per block
#        all_idc = np.arange(len(X))
#        if test:
#            idc,test_idc,_,_ = train_test_split(all_idc,y,test_size = 0.2)
#            splits = list(cv.split(idc, y[idc]))
#            for f,(train_idc, val_idc) in enumerate(splits):
#                splits[f] = (idc[train_idc], idc[val_idc], test_idc)
#        else:
#            splits = list(cv.split(X, y))
#            for f,(train_idc, val_idc) in enumerate(splits):
#                splits[f] = (train_idc, val_idc, None)
#
#        ranks = []
#        for b in range(self.max_blocks):
#            if self.verbose:
#                print(f"Model selection block {b+1}/{self.max_blocks}...")
#            
#            # Evaluate different ranks
#            eval_kwargs = []
#            for fold, (train_idc, val_idc, test_idc) in enumerate(splits):
#                if b:
#                    Xt = self.transform(
#                        X, blocks=[fb[fold] for fb in fold_blocks], select=False
#                    )
#                    err = fold_err[fold]
#                else:
#                    Xt = tl.zeros((n_samples, 0))
#                    err = copy(X)
#                for r in rank_grid:
#                    eval_kwargs.append(dict(Xt=Xt, err=err, y=y, b=b, fold=fold, rank=r, train_idc=train_idc, val_idc=val_idc, test_idc=test_idc))
#            results = Parallel(n_jobs=self.n_jobs)(delayed(self._eval_ranks)(**kwargs) for kwargs in eval_kwargs )
#
#            # Evaluate different n_features
#            eval_kwargs = []
#            for res in results:
#                max_n_features = res['Xt'].shape[-1]
#                if self.select:
#                    n_feature_grid = list(
#                        2 ** np.arange(np.floor(np.log2(max_n_features) + 1), dtype=int)
#                    )
#                    n_feature_grid.append(max_n_features)
#                    n_feature_grid = sorted(list(set(n_feature_grid)))
#
#                    #n_feature_grid = list(range(1,max_n_features+1))
#                else: 
#                    n_feature_grid = [max_n_features]
#
#                for n_features in n_feature_grid:
#                    kwargs = dict(res)
#                    kwargs['n_features'] = n_features
#                    eval_kwargs.append(kwargs)
#            results = Parallel(n_jobs=self.n_jobs)(delayed(self._eval_n_features)(**kwargs) for kwargs in eval_kwargs )
#            self.model_select_info_ += results
#
#            # Select optimal hyperparameters
#            df = pd.DataFrame(results)
#            df = df.groupby(["rank", "n_features"])["val_score"].aggregate("mean")
#            best_rank, best_n_features = df.idxmax()
#            ranks.append(best_rank)
#            # Retrieve matching fold models
#            df = pd.DataFrame(results)
#            df = df.set_index(["rank", "n_features", "fold"])
#            df = df.sort_index()
#            best_blocks = df.loc[(best_rank, best_n_features), "hoda"].tolist()
#
#            # Calculate per fold deflation residuals
#
#            forward_args = []
#            for fold, (train_idc, _,_) in enumerate(splits):
#                block =best_blocks[fold]
#                if b:
#                    err = fold_err[fold]
#                else:
#                    err = copy(X)
#                forward_args.append((block, err,y,train_idc))
#            forward_res = Parallel(n_jobs=self.n_jobs)(delayed(self._fold_forward)(*args) for args in forward_args)
#            curr_fold_blocks, fold_err = zip(*forward_res)
#
#            fold_blocks.append(curr_fold_blocks)
#
#        self.ranks = ranks
#        self.model_select_info_ = pd.DataFrame(self.model_select_info_)
#        self.model_select_info_ = self.model_select_info_.set_index(
#            ["block", "rank", "n_features"]
#        )
#
#        # Truncate and determine best features
#        if self.truncate:
#            df = self.model_select_info_best_
#            df = df.groupby(["block", "rank", "n_features"])
#            df = df["val_score"].aggregate("mean")
#            best_block, _, best_n_features = df.idxmax()
#            best_n_blocks = best_block + 1
#            ranks = ranks[:best_n_blocks]
#
#        # Train BTTDA
#        self.ranks = ranks
#        if self.verbose:
#            print(f"Selected model with ranks {self.ranks}")
#        super().fit(X, y)
#        # Train select
#        Xt = tl.to_numpy(self.transform(X, select=False))
#        self.select_ = SelectKBest(k=best_n_features)
#        self.select_.fit(Xt, y)
#        return self
#
#    @property
#    def model_select_info_best_(self):
#        df_agg = self.model_select_info_.groupby(["block", "rank", "n_features"])
#        df_agg = df_agg['val_score'].aggregate("mean")
#        idc = df_agg.groupby("block").idxmax()
#        df_select = self.model_select_info_.loc[idc]
#        return df_select
#
#    def transform(self, X, select=True, **kwargs):
#        Xt = super().transform(X, **kwargs)
#        if self.select and select:
#            if self.verbose:
#                print(f"Selecting {self.select_.k} features")
#            Xt = self.select_.transform(tl.to_numpy(Xt))
#        return Xt
#
#    def _eval_ranks(self, Xt=None, err=None, y=None, b=None, fold=None, rank=None, train_idc=None, val_idc=None, test_idc=None):
#        n_samples = len(y)
#        hoda_params = self.hoda_params
#        if hoda_params is None:
#            hoda_params = hoda_params
#        hoda = HODA(**hoda_params)
#        hoda.set_params(rank=rank)
#        hoda.fit_backward(err[train_idc], y[train_idc])
#        Xtb = hoda.transform(err)
#        Xtb = tl.reshape(Xtb, (n_samples, -1))
#        Xt = tl.concatenate([Xt, Xtb], axis=1)
#        Xt = tl.to_numpy(Xt)
#        zscore = StandardScaler()
#        zscore.fit(Xt[train_idc], y[train_idc])
#        Xt=zscore.transform(Xt)
#        res = dict(
#            Xt=Xt,
#            y=y,
#            b=b,
#            fold=fold,
#            rank=rank,
#            train_idc=train_idc,
#            val_idc=val_idc,
#            test_idc=test_idc,
#            hoda=hoda,
#        )
#        return res
#
#    def _eval_n_features(self, Xt=None,y=None,b=None,fold=None,rank=None,n_features=None,hoda=None,train_idc=None,val_idc=None, test_idc=None):
#            select = SelectKBest(k=n_features)
#            select.fit(Xt[train_idc], y[train_idc])
#            Xt_sel = select.transform(Xt)
#            clf = LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")
#            clf.fit(Xt_sel[train_idc], y[train_idc])
#            y_pred = clf.decision_function(Xt_sel)
#            train_score = roc_auc_score(y[train_idc], y_pred[train_idc])
#            val_score = roc_auc_score(y[val_idc], y_pred[val_idc])
#            res = dict(
#                    block=b,
#                    fold=fold,
#                    rank=rank,
#                    n_features=n_features,
#                    hoda=hoda,
#                    train_score=train_score,
#                    val_score=val_score,
#            )
#            if test_idc is not None:
#                test_score = roc_auc_score(y[test_idc], y_pred[test_idc])
#                res['test_score'] = test_score
#            return res
# 
#    def _fold_forward(self, block, err,y, train_idc):
#        block.fit_forward(err[train_idc], y[train_idc])
#        err = err- block.inv_transform(block.transform(err))
#        return block,err
 
