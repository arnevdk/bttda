import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import seaborn as sns
import tensorly as tl
from line_profiler_pycharm import profile
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import shrunk_covariance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.utils.extmath import randomized_svd


class HODA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_iter=100,
        tol=1e-13,
        toeplitz=1,
        taper=1,
        rank=None,
        initialize="identity",
        shrinkage="oas",
        verbose=False,
        solver="eig",
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.toeplitz = toeplitz
        self.taper = taper
        self.rank = rank
        self.initialize = initialize
        self.shrinkage = shrinkage
        self.verbose = verbose
        self.solver = solver

    @profile
    def fit(self, X, y):
        X_orig = X.copy()
        # Calculate properties
        X = tl.tensor(X)
        dtype = X.dtype
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        class_size = np.zeros(len(self.classes_), dtype=int)
        for c, cls in enumerate(self.classes_):
            class_size[c] = np.count_nonzero(y == cls)
        n_samples = X.shape[0]
        shape = X.shape[1:]
        order = len(shape)
        self.rank_ = self.rank
        if self.rank_ is None:
            self.rank_ = [0] * order
            for k in range(order):
                prod = 1
                for k2 in range(order):
                    if k != k2:
                        prod *= shape[k2]
                self.rank_[k] = min(shape[k], (n_classes - 1) * prod)
        if self.toeplitz is not None and (self.toeplitz >= order or self.toeplitz < 0):
            raise ValueError(f"Toeplitz mode must bu between 0 and {self.order - 1}")
        if self.taper is not None and (self.taper >= order or self.taper < 0):
            raise ValueError(f"Taper mode must bu between 0 and {self.order - 1}")

        # Construct tapers
        tapers = [None] * order
        for k in range(order):
            if k == self.taper:
                taper = np.linspace(1, 0, num=shape[k])
                tapers[k] = np.zeros((shape[k], shape[k]), dtype=dtype)
                for i in range(shape[k]):
                    taper_vec = np.repeat(taper[i], shape[k] - i)
                    tapers[k] += np.diag(taper_vec, i)
                    if i != 0:
                        tapers[k] += np.diag(taper_vec, -i)
            else:
                tapers[k] = np.ones((shape[k], shape[k]), dtype=dtype)

        # Calculate means and center
        self.class_means_ = tl.zeros((n_classes, *shape), dtype=dtype)
        for i, cls in enumerate(self.classes_):
            self.class_means_[i] = tl.mean(X[y == cls], axis=0)
            X[y == cls] -= self.class_means_[i]
        self.mean_ = tl.mean(self.class_means_, axis=0)

        # Initialize projections
        self.projs_ = [None] * order
        for k in range(order):
            if self.initialize == "identity":
                self.projs_[k] = tl.eye(shape[k], self.rank_[k], dtype=dtype)
            elif self.initialize == "random":
                self.projs_[k] = np.random.rand(shape[k], self.rank_[k])
                if np.iscomplexobj(X):
                    self.projs_[k] = self.projs_[k].astype(dtype)
                    self.projs_[k] += np.random.rand(shape[k], self.rank_[k]) * 1j

            elif self.initialize == "svd":
                Xt = tl.unfold(X, k + 1).T
                _, _, Vt = scipy.sparse.linalg.svds(
                    Xt, k=self.rank[k], return_singular_vectors="vh"
                )
                self.projs_[k] = Vt.T
            else:
                raise ValueError("initialize should be one of {identity, random, svd}")

        # Find projections
        self.updates_ = np.zeros((order, self.max_iter), dtype=float)
        self.objective_ = np.zeros((self.max_iter), dtype=float)
        self.scatter_w_ = [None] * order
        self.scatter_b_ = [None] * order
        for self.iter_ in range(self.max_iter):
            if self.verbose:
                print(f"Iteration {self.iter_}", end="  ")
            new_projs = [None] * order
            for k in range(order):
                X_proj = self._project(X, k)

                # Calculate whithin class scatter

                X_proj = tl.base.partial_unfold(X_proj, k, skip_begin=1)
                X_proj_H = X_proj.conj().transpose((1, 0, 2))
                scatter_w = np.zeros((shape[k], shape[k]), dtype=dtype)
                for j in range(X_proj.shape[-1]):
                    scatter_w += X_proj_H[:, :, j] @ X_proj[:, :, j]

                """
                X_proj = tl.base.unfold(X_proj, k+1)
                scatter_w = X_proj @ X_proj.conj().T
                scatter_w /= np.trace(scatter_w) / shape[k]
                """
                """
                modes = [k2 for k2 in range(order+1) if k2!=k+1]
                scatter_w = tl.tenalg.tensordot(X_proj, X_proj.conj(), modes=modes)
                """
                if self.toeplitz is not None and self.toeplitz == k:
                    scatter_w = force_toeplitz(scatter_w)
                    scatter_w = scipy.linalg.toeplitz(scatter_w)
                if self.taper is not None and self.toeplitz == k:
                    scatter_w *= tapers[k]
                if self.shrinkage == "oas":
                    shrinkage = oas(scatter_w, n_samples)
                else:
                    shrinkage = 0
                mu = np.trace(scatter_w) / shape[k]
                scatter_w = (1 - shrinkage) * scatter_w + shrinkage * mu * np.identity(
                    shape[k]
                )
                self.scatter_w_[k] = scatter_w

                # Calculate between class scatter
                class_means_proj = self._project(self.class_means_, k)
                mean_proj = self._project(self.mean_[np.newaxis], k).squeeze()
                class_means_proj -= mean_proj[np.newaxis]
                class_means_proj = tl.base.partial_unfold(
                    class_means_proj, k, skip_begin=1
                )
                scatter_b = np.zeros((shape[k], shape[k]), dtype=dtype)
                for c in range(n_classes):
                    scatter_b += (
                        class_size[c]
                        * class_means_proj[c]
                        @ class_means_proj[c].conj().T
                    )
                self.scatter_b_[k] = scatter_b

                if self.solver == "eig":
                    # Calculate new projection by solving generalized eigenvalue
                    # problem
                    _, Uk = scipy.linalg.eigh(
                        scatter_b,
                        scatter_w,
                        subset_by_index=[shape[k] - self.rank_[k], shape[k] - 1],
                    )

                elif self.solver == "svd":
                    # Uk, _, _ = randomized_svd(
                    #    scipy.linalg.pinvh(scatter_w) @ scatter_b,
                    #    n_components=self.rank_[k],
                    # )
                    Uk, _, _ = scipy.linalg.svd(
                        scipy.linalg.pinvh(scatter_w) @ scatter_b
                    )
                    Uk = Uk[:, : self.rank_[k]]
                elif self.solver == "seig":
                    _, Uk = scipy.linalg.eigh(
                        scipy.linalg.pinvh(scatter_w) @ scatter_b,
                        subset_by_index=[shape[k] - self.rank_[k], shape[k] - 1],
                    )
                # Orthonormalize for stability
                Uk, _ = scipy.linalg.qr(Uk, mode="economic")
                new_projs[k] = Uk

            # Stopping criterion
            break_flag = True
            for k in range(order):
                tol = self.tol * np.prod(self.projs_[k].shape)
                self.updates_[k, self.iter_] = np.abs(
                    scipy.linalg.norm(new_projs[k] - self.projs_[k], ord="fro")
                )
                if not self.updates_[k, self.iter_] < tol:
                    break_flag = False
            self.projs_ = new_projs
            self.objective_[self.iter_] = self.fisher_ratio(self.transform(X_orig), y)
            if self.verbose:
                print(f"updates: {self.updates_[:, self.iter_]}", end="  ")
                print(f"objective: {self.objective_[self.iter_]}")
            if break_flag:
                break
        return self

    def fisher_ratio(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        dtype = X.dtype
        class_size = np.zeros(len(self.classes_), dtype=int)
        for c, cls in enumerate(self.classes_):
            class_size[c] = np.count_nonzero(y == cls)
        n_samples, n_features = X.shape
        class_means = tl.zeros((n_classes, n_features), dtype=dtype)
        for i, cls in enumerate(classes):
            class_means[i] = np.mean(X[y == cls], axis=0)
            X[y == cls] -= class_means[i][np.newaxis]
        mean = np.mean(class_means, axis=0)
        scatter_w = np.sum(scipy.linalg.norm(X, axis=1))
        class_means -= mean[np.newaxis]
        scatter_b = np.sum(scipy.linalg.norm(class_means, axis=1) * class_size)
        return scatter_b / scatter_w

    @profile
    def _project(self, X, k):
        order = len(X.shape) - 1
        return tl.tenalg.multi_mode_dot(
            X, self.projs_, modes=range(1, order + 1), skip=k, transpose=True
        )
        """
        order = len(X.shape) - 1
        X_proj = X
        for k2 in range(order):
            if k2 != k:
                U = self.projs_[k2]
                X_proj = tl.tenalg.mode_dot(X_proj, U, mode=1 + k2, transpose=True)
        return X_proj
        """

    def transform(self, X, y=None):
        order = len(X.shape) - 1
        X_trans = tl.tenalg.multi_mode_dot(
            X, self.projs_, modes=range(1, order + 1), transpose=True
        )
        """
        order = len(X.shape) - 1
        X_trans = X.copy()
        for k in range(order):
            U = self.projs_[k]
            X_trans = tl.tenalg.mode_dot(X_trans, U, mode=1 + k, transpose=True)
        """
        X_trans = X_trans.reshape(X.shape[0], -1)
        if np.iscomplexobj(X_trans):
            X_trans = np.hstack([np.real(X_trans), np.imag(X_trans)]).astype(np.float64)
        return X_trans


@profile
def force_toeplitz(cov):
    """Coerce the calculated empirical covariance to a Toeplitz-structured
    matrix by setting each diagonal to its mean value.
    """

    n_features = cov.shape[0]
    toeplitz = np.zeros_like(cov[:, 0])
    for i in range(n_features):
        toeplitz[i] = np.mean(np.diag(cov, k=i))
    return toeplitz


@profile
def oas(cov, n_epochs):
    """Calculate the Oracle Approximating Shrinkage coefficient

    Oracle Approximating Shrinkage for empirical covariance matrices [1] or
    Kronecker factor covariance matrices [2].

    [1] Y. Chen, A. Wiesel, Y. C. Eldar, and A. O. Hero, “Shrinkage algorithms
    for MMSE covariance estimation,” IEEE Transactions on Signal Processing,
    vol. 58, no. 10, Art. no. 10, Oct. 2010, doi: 10.1109/tsp.2010.2053029.

    [2] L. Xie, Z. He, J. Tong, T. Liu, J. Li, and J. Xi, “Regularized
    Estimation of Kronecker-Structured Covariance Matrix.” 2021.

    Parameters
    ----------
    cov : array-like of shape (n_features, n_features)
        The empirical covariance matrix before shrinkage.

    n_epochs : int
        The number of epochs.

    Returns
    -------
    shrinkage : float such that 0 <= shrinkage <= 1
        The shrinkage coefficient calculated by Oracle Approximating Shrinkage.
    """
    n_features = cov.shape[0]
    tr_cov = np.trace(cov)
    cov2 = cov.dot(cov)
    tr_cov2 = np.trace(cov2)

    num = tr_cov**2 + (1 - 2 / n_features) * tr_cov2
    den = (
        1 - n_epochs / n_features - (2 * n_epochs) / n_features**2
    ) * tr_cov**2 + (n_epochs + 1 + (2 * (n_epochs - 1)) / n_features) * tr_cov2
    shrinkage = num / den
    shrinkage = min(max(shrinkage, 0), 1)
    # print(shrinkage)
    return shrinkage
