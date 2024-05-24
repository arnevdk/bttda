import math
import pdb

import numpy as np
import tensorly as tl
import tensorly.decomposition
import tensorly.tenalg
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
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
        init="svd",
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
        ortho=False,
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
        self.ortho = ortho
        self.forward = forward

    def fit(self, X, y, classes=None, class_counts=None):
        # Convert to tensor
        if not tl.is_tensor(X):
            X = tl.tensor(X)
        # Fit backward model
        self.fit_backward(X, y, classes=None, class_counts=class_counts)
        # Fit forward model
        if self.forward:
            self.fit_forward(X, y)
        return self

    def fit_backward(
        self, X, y, X_centered=None, means=None, classes=None, class_counts=None
    ):
        # TODO: calculate train info for initialization
        # Convert to tensor
        if not tl.is_tensor(X):
            X = tl.tensor(X)
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
        if self.ortho:
            scatter_x = [None] * order
            for k in range(order):
                scatter_w, _ = mode_scatter(
                    X,
                    k,
                    assume_centered=True,
                    shrinkage=self.shrinkage,
                    toeplitz=self.toeplitz,
                    taper=self.taper,
                )
                scatter_b, _ = mode_scatter(
                    self.means_, k, weights=tl.sqrt(class_counts), shrinkage=0
                )
                scatter_x[k] = scatter_w + scatter_b

        # Iteratively find projections
        iterator = range(1, self.max_iter + 1)
        if self.verbose:
            iterator = tqdm(iterator)
            iterator.set_description("Backward model")
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
                    means_centered_proj, k, weights=tl.sqrt(class_counts), shrinkage=0
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
                    u = u[:, tl.cumsum(w) / tl.sum(w) > self.delta]
                new_rank = u.shape[-1]
                if self.ortho:
                    if self.solver == "lobpcg":
                        solver_params["init"] = u
                    u, w = trunc_eigh(
                        u @ u.T @ (scatter_x[k]) @ u @ u.T,
                        # rank=self.rank_[k],
                        rank=new_rank,
                        method=self.solver,
                        largest=largest,
                        solver_params=solver_params,
                    )

                # A = tl.random.base.random_tensor(
                #    (shape[k], shape[k]), random_state=self.random_state
                # )
                # A, _ = tl.qr(A, mode="reduced")
                # Xk = tl.unfold(X_centered_proj, k + 1)
                # for j in range(100):
                #    en = ElasticNet()
                #    a = u.T @ Xk
                #    b = A @ Xk
                #    en.fit(tl.to_numpy(a.T), tl.to_numpy(b.T))
                #    u_new = tl.to_numpy(en.coef_)
                #    d,_,

                # Calculate update and check convergence
                if u.shape[-1] != self.weights_[k].shape[-1]:
                    update = np.inf
                else:
                    update = tl.metrics.regression.MSE(u, self.weights_[k])
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

    def fit_forward(self, X, y, Xt=None):
        # TODO: add regularization
        # Convert to tensor
        if not tl.is_tensor(X):
            X = tl.tensor(X)
        n_samples, *shape = X.shape
        order = len(shape)

        # Project
        if Xt is None:
            Xt = self.transform(X)

        _, Xt = center(Xt, y)
        _, X = center(X, y)

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
                    G = tl.tenalg.multi_mode_dot(Xt, self.aps_, modes=modes, skip=k)
                    modes = [0] + [kk + 1 for kk in range(order) if kk != k]
                    cov_cross = tl.tensordot(X, G.conj(), axes=(modes, modes))
                    cov_g, shrink = mode_scatter(
                        G,
                        k,
                        # shrinkage=self.shrinkage,
                        shrinkage=0,
                        assume_centered=True,
                    )
                    ap = tl.solve(cov_g.T, cov_cross.T).T

                    # Gk = tl.tenalg.multi_mode_dot(Xt, self.aps_, modes=modes, skip=k)
                    # gk = tl.unfold(Gk, k + 1)
                    # xk = tl.unfold(X, k + 1)

                    # ap, *_ = lstsq(gk.T, xk.T, rcond=shape[k] * np.finfo(X.dtype).eps)
                    ## ap = lstsq_ridge(gk.T, xk.T, lambda_=1)
                    # ap = ap.T

                    update = tl.metrics.regression.MSE(
                        ap / tl.norm(self.aps_[k]), self.aps_[k] / tl.norm(self.aps_[k])
                    )
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
            rank = [min(shape[k], self.rank) for k in range(order)]
        elif self.rank is None:
            rank = shape
        else:
            rank = self.rank

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
                else:
                    raise NotImplementedError
        for k in range(order):
            self.weights_[k] = self.weights_[k] / tl.norm(self.weights_[k])

    def transform(self, X, y=None):
        if not tl.is_tensor(X):
            X = tl.tensor(X)
        order = len(X.shape) - 1
        Xt = tl.tenalg.multi_mode_dot(
            X, self.weights_, modes=range(1, order + 1), transpose=True
        )
        return Xt

    def inv_transform(self, Xt, y=None):
        if not tl.is_tensor(Xt):
            Xt = tl.tensor(Xt)
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
        info = dict()
        info["mse"] = float(mse)
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

    def fit(self, X, y=None, blocks=None):
        if not tl.is_tensor(X):
            X = tl.tensor(X)
        n_samples, *shape = X.shape

        self.classes_, class_counts = np.unique(y, return_counts=True)
        class_counts = tl.tensor(class_counts)

        hoda_params = self.hoda_params or dict()
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
                block, block_info = self._fit_block(
                    X, err, y, rank, hoda_params, class_counts
                )
            self.blocks_.append(block)
            G = block.transform(err)
            err -= block.inv_transform(G)
            # Calculate train info
            train_info_row = dict()
            train_info_row["block"] = self.n_blocks_
            train_info_row["rank"] = block.rank_
            train_info_row.update(block_info)
            if self.extra_train_info:
                train_info_row.update(self._extra_train_info(X, y, class_counts))
            self.train_info_.append(train_info_row)

        # Convert train info to list dict
        self.train_info_ = {
            k: [dic[k] for dic in self.train_info_] for k in self.train_info_[0]
        }

        return self

    def _fit_block(self, _, err, y, rank, hoda_params, class_counts):
        hoda_params["rank"] = rank
        hoda_params["forward"] = True
        block = HODA(**hoda_params)
        block.fit(err, y, classes=self.classes_, class_counts=class_counts)
        return block, dict()

    @property
    def n_blocks_(self):
        return len(self.blocks_)

    @property
    def n_params_(self):
        return sum([b.n_params_ for b in self.blocks_])

    def transform(self, X, y=None, n_blocks=None):
        if not tl.is_tensor(X):
            X = tl.tensor(X)
        err = copy(X)
        n_samples, *_ = X.shape
        Xt = []
        if n_blocks is None:
            n_blocks = self.n_blocks_
        for b in range(n_blocks):
            block = self.blocks_[b]
            Xtb = block.transform(err, y)
            Xt.append(Xtb.reshape(n_samples, -1))
            err -= block.inv_transform(Xtb)

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
        Xt = self.transform(X)
        X_rec = self.inv_transform(Xt)
        mse = tl.metrics.regression.MSE(X_rec, X)
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
        info["F_tr"] = float(F_tr)
        info["F_rt"] = float(F_rt)
        return info


class InfoBTTDA(BTTDA):
    def __init__(
        self,
        ranks=None,
        hoda_params=None,
        extra_train_info=False,
        verbose=False,
        info_crit="aic",
        truncate=False,
    ):
        super().__init__(
            ranks=ranks,
            hoda_params=hoda_params,
            extra_train_info=extra_train_info,
            verbose=verbose,
        )
        self.info_crit = info_crit
        self.truncate = truncate

    def fit(self, X, y=None, blocks=None):
        res = super().fit(X, y=y, blocks=blocks)
        if self.truncate:
            best_n_blocks = np.argmin(self.train_info_[self.info_crit]) + 1
            self.blocks_ = self.blocks_[:best_n_blocks]
        return res

    def _fit_block(self, X, err, y, _, hoda_params, class_counts):
        n_samples, *shape = X.shape

        if self.n_blocks_:
            Xt = self.transform(X)
        ranks = list(2 ** np.arange(np.floor(np.log2(min(shape)) + 1), dtype=int))
        # ranks = [1]
        # ranks = [2]
        # ranks.append(max(shape))
        # ranks = list(range(1, min(shape) + 1))
        best_block = None
        best_Xtb = None
        best_info_crit = np.inf

        for r in ranks:
            hoda = HODA(**hoda_params)
            hoda.set_params(rank=r)
            hoda.fit_backward(err, y)
            Xtb = hoda.transform(err)
            Xtt = Xtb.reshape((n_samples, -1))
            if self.n_blocks_:
                Xtt = tl.concatenate([Xt, Xtt], axis=1)
            clf = self._clf_pipe()
            clf.fit(Xtt, y)
            y_pred = clf.predict_proba(Xtt)
            log_like = -log_loss(y, y_pred, normalize=False)
            n = n_samples
            k = clf[-1].n_features_in_
            if self.info_crit == "bic":
                info_crit = k * tl.log(n) - 2 * log_like
            else:
                aic = 2 * k - 2 * log_like
                if self.info_crit == "aic":
                    info_crit = aic
                elif self.info_crit == "aicc":
                    info_crit = aic + (2 * k**2 + 2 * k) / (n - k - 1)
                else:
                    raise ValueError
            info_crit = float(info_crit)
            if info_crit < best_info_crit:
                best_info_crit = info_crit
                best_block = hoda
                best_Xtb = Xtb

        best_block.fit_forward(err, y, Xt=best_Xtb)
        info = dict()
        info[self.info_crit] = best_info_crit
        return best_block, info

    def _clf_pipe(self):
        return make_pipeline(
            Vectorize(),
            StandardScaler(),
            # SelectF(alpha=0.95),
            LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr"),
        )


class GreedyBTTDA(BTTDA):
    def __init__(
        self,
        ranks=None,
        hoda_params=None,
        extra_train_info=False,
        verbose=False,
        cv=None,
    ):
        super().__init__(
            ranks=ranks,
            hoda_params=hoda_params,
            extra_train_info=extra_train_info,
            verbose=verbose,
        )
        self.cv = cv

    def fit(self, X, y=None):
        n_samples, *shape = X.shape
        if self.cv is None:
            self.cv = StratifiedKFold(shuffle=True, random_state=42)
        self._fold_blocks_ = []
        for f in range(self.cv.n_splits):
            self._fold_blocks_.append([])
        self.rank_grid_ = list(
            2 ** np.arange(np.floor(np.log2(min(shape)) + 1), dtype=int)
        )
        # self.rank_grid_ = [1]
        return super().fit(X, y=y)

    def _fit_block(self, X, err, y, _, hoda_params, class_counts):
        n_samples, *shape = X.shape

        clf = self._clf_pipe()
        train_scores = tl.zeros((self.cv.n_splits, len(self.rank_grid_)))
        val_scores = tl.zeros((self.cv.n_splits, len(self.rank_grid_)))
        # test_scores = tl.zeros((self.cv.n_splits, len(self.rank_grid_)))
        fold_blocks = []
        # splits must stay consistent over blocks
        for fold, (train_idc, val_test_idc) in enumerate(self.cv.split(err, y)):
            # test_idc = val_test_idc[: len(val_test_idc) // 2]
            # val_idc = val_test_idc[len(val_test_idc) // 2 :]
            val_idc = val_test_idc
            rank_blocks = []
            for ri, r in enumerate(self.rank_grid_):
                hoda = HODA(**hoda_params)
                hoda.set_params(rank=r)
                hoda.fit(err[train_idc], y[train_idc])
                Xtb = hoda.transform(err)
                Xtb = tl.reshape(Xtb, (n_samples, -1))
                if self.n_blocks_:
                    Xt = self._fold_transform(X, fold)
                    # Xt = self.transform(X)
                    Xtb = tl.concatenate([Xt, Xtb], axis=1)
                clf.fit(Xtb[train_idc], y[train_idc])
                y_pred = clf.decision_function(Xtb)
                train_scores[fold, ri] = roc_auc_score(y[train_idc], y_pred[train_idc])
                val_scores[fold, ri] = roc_auc_score(y[val_idc], y_pred[val_idc])
                # test_scores[fold, ri] = roc_auc_score(y[test_idc], y_pred[test_idc])
                rank_blocks.append(hoda)
            fold_blocks.append(rank_blocks)
        rank_train_scores = tl.mean(train_scores, axis=0)
        rank_val_scores = tl.mean(val_scores, axis=0)
        # rank_test_scores = tl.mean(test_scores, axis=0)
        best_rank_idx = int(tl.argmax(rank_val_scores))
        for f in range(self.cv.n_splits):
            best_fold_block = fold_blocks[f][best_rank_idx]
            best_fold_block.fit_forward(err, y)
            self._fold_blocks_[f].append(best_fold_block)
        train_score = float(rank_train_scores[best_rank_idx])
        val_score = float(rank_val_scores[best_rank_idx])
        # test_score = float(rank_test_scores[best_rank_idx])
        best_rank = int(self.rank_grid_[best_rank_idx])
        best_block = HODA(**hoda_params)
        best_block.set_params(rank=best_rank, forward=True)
        best_block.fit(err, y)
        info = dict(
            rank=best_rank,
            train_score=train_score,
            val_score=val_score,
            # test_score=test_score,
            train_scores=tl.to_numpy(val_scores),
            val_scores=tl.to_numpy(val_scores),
            # test_scores=tl.to_numpy(val_scores),
        )
        return best_block, info

    def _fold_transform(self, X, fold, y=None, n_blocks=None):
        if not tl.is_tensor(X):
            X = tl.tensor(X)
        err = copy(X)
        n_samples, *_ = X.shape
        Xt = []
        if n_blocks is None:
            n_blocks = self.n_blocks_
        for b in range(n_blocks):
            block = self._fold_blocks_[fold][b]
            Xtb = block.transform(err, y)
            Xt.append(Xtb.reshape(n_samples, -1))
            err -= block.inv_transform(Xtb)

        Xt = tl.concatenate(Xt, axis=1)
        return Xt

    def _clf_pipe(self):
        return make_pipeline(
            Vectorize(),
            StandardScaler(),
            # SelectF(alpha=0.95),
            LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr"),
        )
