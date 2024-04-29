import math
import pdb

import numpy as np
import tensorly as tl
import tensorly.decomposition
import tensorly.tenalg
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm

from hoda.backend import copy, lstsq
from hoda.classification import SelectF, Vectorize
from hoda.cov import KroneckerCovariance, mode_scatter
from hoda.hopls import TOT
from hoda.util import center, f_multiway, trunc_eigh


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
        fit_forward=False,
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
        self.fit_forward = fit_forward

    def fit(self, X, y):
        # Convert to tensor
        if not tl.is_tensor(X):
            X = tl.tensor(X)

        # Initialize s
        self.train_info_ = []

        # Determine classses  and center
        self.classes_, class_counts = np.unique(y, return_counts=True)
        class_counts = tl.tensor(class_counts)
        self.means_, X_centered = center(X, y, self.classes_)

        # Initialize backward projections
        self._init_backward(X_centered, self.rank, self.init)
        # Fit backward model
        self._fit_backward(X, X_centered, y, class_counts)

        # Initialize forward model
        if self.fit_forward:
            Xt = self.transform(X)
            _, Xt_centered = center(Xt, y, self.classes_)
            self._init_forward(X, Xt, X_centered, Xt_centered, y)
            # Fit forward model
            self._fit_forward(X, Xt, X_centered, Xt_centered, y)

        # Convert train_info to list dict
        self.train_info_ = {
            k: [dic[k] for dic in self.train_info_] for k in self.train_info_[0]
        }
        return self

    @property
    def rank_(self):
        order = len(self.weights_)
        return tuple([self.weights_[k].shape[-1] for k in range(order)])

    def _init_backward(self, X_centered, rank, method):
        _, *shape = X_centered.shape
        order = X_centered.ndim - 1

        # Determine multilinear rank
        modes = [k + 1 for k in range(order)]
        if isinstance(rank, np.integer) or isinstance(rank, int):
            rank = [min(shape[k], self.rank) for k in range(order)]
        elif self.rank is None:
            rank = shape
        else:
            rank = self.rank

        if method == "mlsvd":
            (_, self.weights_), _ = tl.decomposition.partial_tucker(
                X_centered,
                rank=rank,
                modes=modes,
            )
        else:
            self.weights_ = [None] * order
            for k in range(order):
                if method == "svd":
                    X_centered_k = tl.unfold(X_centered, k + 1)
                    self.weights_[k], _, _ = tl.tenalg.svd_interface(
                        X_centered_k, method="truncated_svd", n_eigenvecs=rank[k]
                    )
                elif method == "random":
                    self.weights_[k] = tl.random.base.random_tensor(
                        (shape[k], rank[k]), random_state=self.random_state
                    )
                    self.weights_[k], _ = tl.qr(self.weights_[k], mode="reduced")
                else:
                    raise NotImplementedError
        for k in range(order):
            self.weights_[k] = self.weights_[k] / tl.norm(self.weights_[k])

    def _fit_backward(self, X, X_centered, y, class_counts):
        n_samples, *shape = X.shape
        order = len(shape)

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
        self.scatter_w_ = [None] * order
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
                    Xt = self.transform(X)
                    train_info_row.update(
                        extra_train_info(
                            X,
                            Xt,
                            y,
                            classes=self.classes_,
                            class_counts=class_counts,
                        )
                    )
                self.train_info_.append(train_info_row)

            # Exit if converged
            if converged:
                break
        pass

    def _init_forward(self, X, Xt, X_centered, Xt_centered, y):
        n_samples, *shape = X.shape
        order = len(shape)

        # cov_G = KroneckerCovariance(estimator="mle")
        # cov_G.fit(Xt_centered, Xt_centered)

        self.aps_ = [None] * order

        for k in range(order):
            modes = range(1, order + 1)
            X_centered_proj = tl.tenalg.multi_mode_dot(
                X_centered,
                self.weights_,
                modes=modes,
                skip=k,
                transpose=True,
            )
            cov_x, _ = mode_scatter(
                X_centered_proj,
                k,
                assume_centered=True,
                shrinkage=self.shrinkage,
                # shrinkage=0,
                # toeplitz=self.toeplitz,
                # taper=self.taper,
            )
            cov_x /= n_samples * math.prod(self.rank_) / self.rank_[k] - 1

            cov_g, shrink_g = mode_scatter(
                Xt_centered,
                k,
                assume_centered=True,
                shrinkage=0,
            )
            cov_g /= n_samples * math.prod(self.rank_) / self.rank_[k] - 1
            # cov_g = cov_G.covs_[k]

            P_inv = tl.diag(1 / tl.diag(cov_g))
            self.aps_[k] = cov_x @ (
                tl.solve(cov_g @ P_inv, self.weights_[k].T).T @ P_inv
            )

    # def _fit_forward(self, X, Xt, X_centered, Xt_centered, y):
    #    """
    #    https://etna.math.kent.edu/vol.55.2022/pp92-111.dir/pp92-111.pdf
    #    HOPLS
    #    """

    # def _fit_forward(self, X, Xt, X_centered, Xt_centered, y):
    #    n_samples, *shape = X.shape
    #    order = len(shape)

    #    self.cov_x_ = [None] * order
    #    self.cov_g_ = [None] * order
    #    self.scale_x_ = tl.zeros(order)
    #    self.scale_g_ = tl.zeros(order)
    #    self.aps_ = [None] * order
    #    for k in range(order):
    #        cov_x, shrink_x = mode_scatter(
    #            X_centered,
    #            k,
    #            assume_centered=True,
    #            # shrinkage=self.shrinkage,
    #            # toeplitz=self.toeplitz,
    #            # taper=self.taper,
    #        )
    #        cov_g, shrink_g = mode_scatter(
    #            Xt_centered,
    #            k,
    #            # shrinkage=self.shrinkage,
    #            assume_centered=True,
    #        )

    #        scale_x = (tl.trace(cov_x) / shape[k]) / (
    #            n_samples * math.prod(shape) / shape[k] - 1
    #        )
    #        scale_g = (tl.trace(cov_g) / self.rank_[k]) / (
    #            n_samples * math.prod(self.rank_) / self.rank_[k] - 1
    #        )

    #        cov_x /= tl.trace(cov_x) / shape[k]
    #        cov_g /= tl.trace(cov_g) / self.rank_[k]

    #        self.cov_g_[k] = cov_g
    #        self.cov_x_[k] = cov_x
    #        self.scale_x_[k] = scale_x
    #        self.scale_g_[k] = scale_g
    #        # self.aps_[k] = cov_x @ tl.solve(cov_g, self.weights_[k].T).T
    #        P_inv = tl.diag(1 / tl.diag(cov_g))
    #        self.aps_[k] = cov_x @ (
    #            tl.solve(cov_g @ P_inv, self.weights_[k].T).T @ P_inv
    #        )

    #    self.scale_x_ = tl.mean(self.scale_x_)
    #    self.scale_g_ = tl.mean(self.scale_g_)

    # def _fit_forward(self, X, Xt, X_centered, Xt_centered, y):
    #    n_samples, *shape = X.shape
    #    order = len(shape)
    #    x_proj_part = [None] * order
    #    for k in range(order):
    #        modes = range(1, order + 1)
    #        x_proj_part[k] = tl.tenalg.multi_mode_dot(
    #            X_centered, self.weights_, modes=modes, skip=k, transpose=True
    #        )

    #    iterator = range(1, self.max_iter + 1)
    #    if self.verbose:
    #        iterator = tqdm(iterator)
    #        iterator.set_description("Forward model ")
    #    converged = False
    #    for self.iter_ in iterator:
    #        if self.extra_train_info:
    #            X_approx = self.inv_transform(Xt)
    #            err = X - X_approx

    #            n_samples = X.shape[0]
    #            G_flat = tl.to_numpy(
    #                Xt.reshape((n_samples, -1), order="F"),
    #            )
    #            err_flat = tl.to_numpy(err.reshape((n_samples, -1), order="F"))
    #            cross_corr = np.corrcoef(G_flat, err_flat, rowvar=False)
    #            cross_corr = cross_corr[G_flat.shape[-1] :, : G_flat.shape[-1]].T
    #            print(
    #                f"MSE: {tl.metrics.regression.MSE(X, X_approx)} corr: {tl.metrics.regression.MSE(cross_corr, 0)}"
    #            )

    #        if converged:
    #            break

    #        converged = True

    #        for k in range(order):
    #            modes = range(1, order + 1)
    #            x = tl.tenalg.multi_mode_dot(
    #                x_proj_part[k], self.aps_, modes=modes, skip=k
    #            )
    #            cov_x, _ = mode_scatter(
    #                x,
    #                k,
    #                assume_centered=True,
    #                # shrinkage=self.shrinkage,
    #                # toeplitz=self.toeplitz,
    #                # taper=self.taper,
    #                shrinkage=0,
    #            )
    #            cov_x /= n_samples * math.prod(shape) / shape[k] - 1

    #            g = tl.tenalg.multi_mode_dot(
    #                Xt_centered, self.aps_, modes=modes, skip=k
    #            )
    #            cov_g, shrink_g = mode_scatter(
    #                g,
    #                k,
    #                assume_centered=True,
    #                # shrinkage=self.shrinkage,
    #                shrinkage=0,
    #            )
    #            cov_g /= n_samples * math.prod(shape) / shape[k] - 1
    #            P_inv = tl.diag(1 / tl.diag(cov_g))
    #            ap = cov_x @ (tl.solve(cov_g @ P_inv, self.weights_[k].T).T @ P_inv)

    #            # ap = cov_x @ (tl.solve(cov_g, self.weights_[k].T).T)
    #            update = tl.metrics.regression.MSE(
    #                ap / tl.norm(self.aps_[k]), self.aps_[k] / tl.norm(self.aps_[k])
    #            )
    #            self.aps_[k] = ap

    #        converged = update < self.tol and converged

    def _fit_forward(self, X, Xt, X_centered, Xt_centered, y):
        # TODO: add regularization
        # TODO: add weighted
        n_samples, *shape = X.shape
        order = len(shape)

        iterator = range(1, self.max_iter + 1)
        if self.verbose:
            iterator = tqdm(iterator)
            iterator.set_description("Forward model ")
        converged = False
        for self.iter_ in iterator:
            if self.extra_train_info:
                X_approx = self.inv_transform(Xt)
                err = X - X_approx

                n_samples = X.shape[0]
                G_flat = tl.to_numpy(
                    Xt.reshape((n_samples, -1), order="F"),
                )
                err_flat = tl.to_numpy(err.reshape((n_samples, -1), order="F"))
                cross_corr = np.corrcoef(G_flat, err_flat, rowvar=False)
                cross_corr = cross_corr[G_flat.shape[-1] :, : G_flat.shape[-1]].T
                print(
                    f"MSE: {tl.metrics.regression.MSE(X, X_approx)} corr: {tl.metrics.regression.MSE(cross_corr, 0)}"
                )

            if converged:
                break

            converged = True

            for k in range(order):
                modes = range(1, order + 1)
                gk = tl.tenalg.multi_mode_dot(Xt, self.aps_, modes=modes, skip=k)
                gk = tl.unfold(gk, k + 1)
                xk = tl.unfold(X, k + 1)

                ap, *_ = lstsq(gk.T, xk.T)
                ap = ap.T

                update = tl.metrics.regression.MSE(
                    ap / tl.norm(self.aps_[k]), self.aps_[k] / tl.norm(self.aps_[k])
                )
                self.aps_[k] = ap

            converged = update < self.tol and converged

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

    # def inv_transform(self, Xt, y=None):
    #    if not tl.is_tensor(Xt):
    #        Xt = tl.tensor(Xt, dtype=Xt.dtype)
    #    return (
    #        tl.tenalg.multi_mode_dot(Xt, self.aps_, modes=[1, 2])
    #        * self.scale_x_
    #        / self.scale_g_
    #    )

    @property
    def n_params_(self):
        return sum([s.size for s in self.weights_])


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
        X = tl.tensor(X)
        n_samples, *shape = X.shape

        self.classes_, class_counts = np.unique(y, return_counts=True)
        class_counts = tl.tensor(class_counts)

        hoda_params = self.hoda_params or dict()
        hoda_params["fit_forward"] = True
        self.blocks_ = []
        self.train_info_ = []

        err = copy(X)
        for b, rank in enumerate(self.ranks):
            hoda_params["rank"] = rank
            if blocks is not None and b < len(blocks):
                # TODO error if ranks are not equal
                block = blocks[b]
            else:
                if self.verbose:
                    print(f"Fitting block {b+1}/{len(self.ranks)}...")
                block = HODA(**hoda_params)
                block.fit(err, y)
            self.blocks_.append(block)
            G = block.transform(err)
            err -= block.inv_transform(G)
            # Store train info
            row = dict()
            row["block"] = self.n_blocks_
            row["rank"] = block.rank_
            if self.extra_train_info:
                Xt = self.transform(X)
                X_rec = self.inv_transform(Xt)
                row.update(
                    extra_train_info(
                        X,
                        Xt,
                        y,
                        X_rec=X_rec,
                        classes=self.classes_,
                        class_counts=class_counts,
                    )
                )

            self.train_info_.append(row)

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


# class InfoBTTDA(BTTDA):
#    def __init__(
#        self,
#        hoda_params=None,
#        extra_train_info=False,
#        verbose=False,
#        max_blocks=8,
#        info_crit="bic",
#    ):
#        super().__init__(
#            hoda_params=hoda_params, extra_train_info=extra_train_info, verbose=False
#        )
#        self.max_blocks = max_blocks
#        self.sub_verbose = verbose
#        self.info_crit = info_crit
#
#    def fit(self, X, y=None, X_test=None, y_test=None, blocks=None):
#        n_samples, *shape = X.shape
#
#        self.blocks_ = []
#        info_crits = []
#        bttda_params = {
#            k: v
#            for k, v in self.get_params().items()
#            if k in BTTDA().get_params().keys()
#        }
#        for b in range(self.max_blocks):
#            if self.sub_verbose:
#                print(f"Fitting block {b+1}/{self.max_blocks}...")
#            ranks = list(2 ** np.arange(np.floor(np.log2(min(shape)) + 1), dtype=int))
#            # ranks = [2]
#            # ranks.append(max(shape))
#            # ranks = list(range(1, min(shape) + 1))
#            best_bttda = None
#            best_info_crit = np.inf
#            for r in ranks:
#                bttda = BTTDA(**bttda_params)
#                params = dict(ranks=(*self.ranks_, r))
#                bttda.set_params(**params)
#                bttda.fit(X, y, blocks=self.blocks_)
#                Xt = bttda.transform(X)
#                # F = f_multiway(Xt, y, method=bttda.blocks_[0].obj)
#                F = f_multiway(Xt, y, method="rt")
#                log_like = tl.log(F) * n_samples
#                # lda = LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")
#                # lda = lda.fit(scipy.stats.zscore(vec(Xt)), y) log_like = -log_loss(y, lda.predict_proba(vec(Xt)), normalize=False)
#                n = n_samples
#                # k = bttda.n_params_
#                k = math.prod(Xt.shape[1:])
#                if self.info_crit == "bic":
#                    info_crit = k * tl.log(n) - 2 * log_like
#                else:
#                    aic = 2 * k - 2 * log_like
#                    if self.info_crit == "aic":
#                        info_crit = aic
#                    elif self.info_crit == "aicc":
#                        info_crit = aic + (2 * k**2 + 2 * k) / (n - k - 1)
#                    else:
#                        raise ValueError
#                info_crit = float(info_crit)
#                if info_crit < best_info_crit:
#                    best_info_crit = info_crit
#                    best_bttda = bttda
#            self.blocks_ = best_bttda.blocks_
#            self.train_info_ = best_bttda.train_info_
#            info_crits.append(best_info_crit)
#            self.train_info_[self.info_crit] = info_crits
#
#        return self


class GreedyBTTDA(BTTDA):
    def __init__(
        self,
        hoda_params=None,
        extra_train_info=False,
        verbose=False,
        max_blocks=8,
        gs_params=None,
        truncate=True,
    ):
        super().__init__(
            hoda_params=hoda_params, extra_train_info=extra_train_info, verbose=False
        )
        self.max_blocks = max_blocks
        self.gs_params = gs_params
        self.sub_verbose = verbose
        self.truncate = truncate

    def fit(self, X, y=None, X_test=None, y_test=None, blocks=None):
        _, *shape = X.shape
        gs_params = self.gs_params or dict()
        gs_params.setdefault("scoring", "roc_auc")
        gs_params["refit"] = True

        self.blocks_ = []
        val_scores = []
        test_scores = []
        for b in range(self.max_blocks):
            if self.sub_verbose:
                print(f"Fitting block {b+1}/{self.max_blocks}...")
            if blocks is not None and len(blocks) <= b:
                self.blocks_ = blocks[:b]
                val_scores.append(np.nan)
                test_scores.append(np.nan)
                continue

            pipe = self._build_clf_pipe()
            # ranks = list(2 ** np.arange(np.floor(np.log2(max(shape))) + 1, dtype=int))
            # ranks = list(range(1, min(shape) + 1))
            # ranks = [1, 2, 3, 4]
            ranks = list(2 ** np.arange(np.floor(np.log2(min(shape))) + 1, dtype=int))
            param_grid = dict(bttda__ranks=[(*self.ranks_, r) for r in ranks])
            gs = GridSearchCV(pipe, param_grid, **gs_params)
            gs.fit(X, y, bttda__blocks=self.blocks_)
            if self.truncate and b and gs.best_score_ < val_scores[-1]:
                break
            bttda = gs.best_estimator_["bttda"]
            self.blocks_ = bttda.blocks_
            self.train_info_ = bttda.train_info_
            val_scores.append(gs.best_score_)
            if X_test is not None:
                test_score = gs.scorer_(gs.best_estimator_, X_test, y_test)
                test_scores.append(test_score)

        self.train_info_["val_score"] = val_scores
        if X_test is not None:
            self.train_info_["test_score"] = test_scores

        return self

    def _build_clf_pipe(self):
        params = {
            k: v
            for k, v in self.get_params().items()
            if k in BTTDA().get_params().keys()
        }
        return make_pipeline(
            BTTDA(**params),
            Vectorize(),
            StandardScaler(),
            SelectF(alpha=0.5),
            LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr"),
        )


def extra_train_info(X, Xt, y, X_rec=None, classes=None, class_counts=None):
    if classes is None or class_counts is None:
        classes, class_counts = np.unique(y, re0turn_counts=True)
    info = dict()
    # Objective: multi-way F-score
    # trace-ratio
    F_tr = f_multiway(
        Xt,
        y,
        classes,
        class_counts,
        method="tr",
    )
    info["F_tr"] = float(F_tr)
    # ratio-trace
    F_rt = f_multiway(
        Xt,
        y,
        classes,
        class_counts,
        method="rt",
    )
    info["F_rt"] = float(F_rt)

    # MSE
    if X_rec is not None:
        mse = tl.metrics.regression.MSE(X, X_rec)
        info["mse"] = float(mse)
    return info
