import math
import warnings

import numpy as np
import tensorly as tl
from numpy.linalg import LinAlgError
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from bttda.cov import mode_scatter
from bttda.util import center, norm_fro, ridge_regression, solve_gevdh

tl.tenalg.set_backend("einsum")
tl.plugins.use_opt_einsum()


def obj_rt(scatter_b, scatter_w, _):
    """Ratio-trace discriminant analysis objective function.

    Parameters
    ----------
    scatter_b : tensorly.tensor of shape (dim_k, dim_k)
        Between-class scatter matrix.

    scatter_w : tensorly.tensor of shape (dim_k, dim_k)
        Within-class scatter matrix.

    _ : ignored
        Additional positional argument for compatibility.

    Returns
    -------
    A : tensorly.tensor of shape (dim_k, dim_k)
        Symmetric matrix A for generalized eigendecomposition.

    B : tensorly.tensor of shape (dim_k, dim_k)
        Symmetric matrix B for generalized eigendecomposition.

    Notes
    -----
    The objective is defined as::

        φ = Tr(Uᵀ S_w⁻¹ S_b U)

    References
    ----------
    [1] Phan, A. H., & Cichocki, A. (2010).
        Tensor decompositions for feature extraction and classification of high
        dimensional datasets. Nonlinear theory and its applications, IEICE,
        1(1), 37–68.

    [2] Wang, H., Yan, S., Xu, D., Tang, X., & Huang, T. (2007, June).
        Trace ratio vs. ratio trace for dimensionality reduction.
        In 2007 IEEE Conference on Computer Vision and Pattern Recognition (pp. 1–8). IEEE.
    """
    return scatter_b, scatter_w


def obj_tr(scatter_b, scatter_w, u, psi=1):
    """Trace-ratio discriminant analysis objective function.

    Parameters
    ----------
    scatter_b : tensorly.tensor of shape (dim_k, dim_k)
        Between-class scatter matrix.

    scatter_w : tensorly.tensor of shape (dim_k, dim_k)
        Within-class scatter matrix.

    u : tensorly.tensor of shape (dim_k, rank_k)

    psi : float, default=1
        Scaling factor for the between-within class scatter difference.
       Initial or previous estimate of the discriminant weights.

    Returns
    -------
    A : tensorly.tensor of shape (dim_k, dim_k)
        Symmetric matrix A for generalized eigendecomposition.

    _ : None
        Additional output for compatibility.

    Notes
    -----
    The objective is defined as::

        φ = Tr(Uᵀ S_b U) / Tr(Uᵀ S_w U)

    References
    ----------
    [1] Phan, A. H., & Cichocki, A. (2010).
        Tensor decompositions for feature extraction and classification of high
        dimensional datasets. Nonlinear theory and its applications, IEICE,
        1(1), 37–68.

    [2] Wang, H., Yan, S., Xu, D., Tang, X., & Huang, T. (2007, June).
        Trace ratio vs. ratio trace for dimensionality reduction.
        In 2007 IEEE Conference on Computer Vision and Pattern Recognition (pp. 1–8). IEEE.
    """
    scatter_b_t = u.T @ scatter_b @ u
    tr_scatter_b_t = tl.trace(scatter_b_t)
    scatter_w_t = u.T @ scatter_w @ u
    tr_scatter_w_t = tl.trace(scatter_w_t)
    phi = tr_scatter_b_t / tr_scatter_w_t
    A = scatter_b - psi * phi * scatter_w
    return A, None


def obj_lfl(scatter_b, scatter_w, u, psi=1):
    """Linear feature learning discriminant objective function.

    Parameters
    ----------
    scatter_b : tensorly.tensor of shape (dim_k, dim_k)
        Between-class scatter matrix.

    scatter_w : tensorly.tensor of shape (dim_k, dim_k)
        Within-class scatter matrix.

    u : tensorly.tensor of shape (dim_k, rank_k)
        Initial or previous estimate of the discriminant weights.

    psi : float, default=1
        Scaling factor for the between-within class scatter difference.

    Returns
    -------
    A : tensorly.tensor of shape (dim_k, dim_k)
        Symmetric matrix A for generalized eigendecomposition.

    B : tensorly.tensor of shape (dim_k, dim_k)
        Symmetric matrix B for generalized eigendecomposition.

    References
    ----------
    [1] Aghili, S. N., Kilani, S., Khushaba, R. N., & Rouhani, E. (2023).
        A spatial–temporal linear feature learning algorithm for P300-based
        brain–computer interfaces. *Heliyon*, 9(4).
    """
    phi = tl.trace(u.T @ scatter_b @ u) / tl.trace(u.T @ scatter_w @ u)
    A = scatter_b - psi * phi * scatter_w
    B = scatter_w
    return A, B


def obj_od(scatter_b, scatter_w, u):
    """Optimal dimensionality discriminant analysis objective function.

    Parameters
    ----------
    scatter_b : tensorly.tensor of shape (dim_k, dim_k)
        Between-class scatter matrix.

    scatter_w : tensorly.tensor of shape (dim_k, dim_k)
        Within-class scatter matrix.

    u : tensorly.tensor of shape (dim_k, rank_k)
        Initial or previous estimate of the discriminant weights.

    Returns
    -------
    A : tensorly.tensor of shape (dim_k, dim_k)
        Symmetric matrix A for eigendecomposition.

    _ : None
        Additional output for compatibility.

    References
    ----------
    [1] Nie, F., Xiang, S., Song, Y., & Zhang, C. (2007, April).
        Extracting the optimal dimensionality for discriminant analysis.
        In *ICASSP'07* (Vol. 2, pp. II-617). IEEE.

    [2] Wang, J., Wang, L., Nie, F., & Li, X. (2021).
        A novel formulation of trace-ratio linear discriminant analysis.
        *IEEE Transactions on Neural Networks and Learning Systems*, 33(10), 5568–5578.
    """
    s = tl.trace(u.T @ scatter_b @ u) / tl.trace(u.T @ scatter_w @ u)
    return -(s**2 * scatter_w - 2 * s * scatter_b), None


def obj_sr(scatter_b, scatter_w, u):
    """Spectral regression discriminant analysis objective function.

    Parameters
    ----------
    scatter_b : tensorly.tensor of shape (dim_k, dim_k)
        Between-class scatter matrix.

    scatter_w : tensorly.tensor of shape (dim_k, dim_k)
        Within-class scatter matrix.

    u : tensorly.tensor of shape (dim_k, rank_k)
        Initial or previous estimate of the discriminant weights.

    Raises
    ------
    NotImplementedError
        This objective function is not yet implemented.

    References
    ----------
    [1] Idaji, M. J., Shamsollahi, M. B., & Sardouie, S. H. (2017).
        Higher order spectral regression discriminant analysis (HOSRDA):
        A tensor feature reduction method for ERP detection.
        *Pattern Recognition*, 70, 152–162.
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
    """Higher-Order Discriminant Analysis (HODA) tensor decomposition method.

    HODA [1] efficiently decomposes a tensor X into a low-dimensional core tensor
    G with per-mode activation matrices Aₖ using per-mode weight matrices Wₖ,
    such that discriminability between given classes is maximal.

    The backward model obtains G from input data X as::

        G = X ×₁ W₁ ×₂ W₂ × ... ×ₖ Wₖ

    The weights Wₖ are obtained by iteratively solving alternating per-mode
    generalized eigendecomposition problems.

    The forward model reconstructs X from G as::

        X ≈ G ×₁ A₁ᵀ ×₂ A₂ᵀ × ... ×ₖ Aₖᵀ

    The activations Aₖ are obtained by iteratively solving an alternating
    least-squares problem.

    This implementation supports multiple discriminant objective functions and
    solvers for the internal generalized eigenvalue problem. Per-mode scatter
    matrices can be regularized using one of the available scatter shrinkage
    methods or by imposing a Toeplitz-matrix structure [2] to improve performance
    when a mode corresponds to an evenly sampled stationary signal.

    Parameters
    ----------
    rank : int, tuple of int, or None, default=None
        Desired dimensionality (rank) of the core tensor.

        If rank is not None, theta must be None. If both theta and rank are None
        rank (dim_1, dim_2, ..., dim_K) is assumed.

    theta : float or None, default=None
        Automatically determines the mode ranks based on the explained proportion
        of the per-mode total scatter matrices.

        Values should be 0 ≤ θ ≤ 1, with 0 corresponding to rank (0, 0, ..., 0)
        and 1 corresponding to rank (dim_1, dim_2, ..., dim_K).
        If theta is not None, rank must be None.

    obj : {'tr', 'rt', 'lfl', 'od', 'sr'}, default='tr'
        Discriminant objective used to construct the generalized eigenvalue problem.
        The following options are available:

        - **'tr'** : *Trace-Ratio* objective
          Maximizes the ratio of between-class to within-class scatter traces:
          φ = Tr(Uᵀ S_b U) / Tr(Uᵀ S_w U).

        - **'rt'** : *Ratio-Trace* objective
          Equivalent to classical Fisher Discriminant Analysis; maximizes
          Tr(Uᵀ S_w⁻¹ S_b U).

        - **'lfl'** : *Linear Feature Learning* objective.
          approximates a mix of trace-ratio and ratio-trace discriminant analysis.

        - **'od'** : *Optimal Dimensionality* objective.
          Automatically determines the most discriminative subspace dimension
          by optimizing a quadratic trace-ratio formulation.

        - **'sr'** : *Spectral Regression* objective.
          (Not yet implemented.)

    shrinkage : {'lw', 'oas', 'ss', 'ell', 'loocv'} or float or tuple of (str or float), default='lw'
        Shrinkage method or factor used to regularize the within-class scatter
        matrix. See `bttda.cov.mode_scatter` for available options and further details.

        - If a **float** between 0.0 and 1.0, the within-class scatter is directly
          regularized with this factor.

        - If a **string**, the shrinkage factor is estimated automatically using
          the specified method.

        - If a **tuple**, it must contain per-mode specifications (e.g.,
          `(lw', 0.3, 'oas')`) allowing mixed shrinkage settings across tensor modes.

    toeplitz : tuple of int or None, default=None
        Modes for which to impose a Toeplitz structure on the within-class scatter matrix.

    solver : {'lanczos', 'lobpcg', 'svd'}, default='lanczos'
        Method for solving the generalized eigenvalue problem.
        'svd' can only be used with objectives yielding symmetric
        positive definite problems (currently only 'rt').

    max_iter : int, default=256
        Maximum number of iterations for backward and forward solving.

    tol : float, default=1e-8
        Convergence tolerance for early stopping when updates are small.

    solver_params : dict or None, default=None
        Additional keyword arguments passed to the eigendecomposition solver.

    verbose : bool, default=False
        If True, print progress information during fitting.

    extra_train_info : bool, default=False
        If True, calculate and store additional statistics (e.g., objective values)
        during iterations. This slows down fitting.

    forward : bool, default=False
        If True, fit the forward model during `fit`. Otherwise, only fit the backward model.

    Attributes
    ----------
    weights_ : list of tensorly.tensor of shape (dim_k, rank_k)
        List of per-mode projection matrices Wₖ, each of size
        ``(dim_k, rank_k)``, where ``dim_k`` is the size of mode *k* of the
        input tensor. The list length equals the tensor order. These matrices
        define the backward (discriminant) model:

            G = X ×₁ W₁ ×₂ W₂ × ... ×ₖ Wₖ

    activation_patterns_: list of tensorly.tensor of shape (dim_k, rank_k)
        List of per-mode reconstruction matrices Aₖ computed when
        the forward model is fitted (i.e., when ``forward=True``). They define
        the reconstruction of X from the core tensor G as:

            X ≈ G ×₁ A₁ᵀ ×₂ A₂ᵀ × ... ×ₖ Aₖᵀ

        Only set when using `fit` when  `forward=True` or when using `fit_foward`.

    classes_: list of obj
        List of length n_classes unique classes occuring in `y`, in increasing order.

    means_ : tensorly.tensor of shape (n_classes, dim_1, dim_2, ..., dim_k)
        Class means.

    train_info_ : dict()
        A dictionary storing statistics gathered during backward and forward
        fitting. `train_info_` contains two entries, **'backward'** and **'forward'**,
        respectively storing information from the backward and forward modeling
        algorithm. These each contain a list of dictionaries for each iteration
        storing key-value pairs for that iteration. Following keys are available,
        if `extra_train_info` is true, keys marked with 'extra' are calculated
        and stored.

        Backward:
            - **'iteration'**:
                The outer loop iteration. A step for each mode is performed
                per iteration.
            - **'mode'**: The mode for which the current statistics are calculated.
            - **'flip'**:
                Each update per iteration and relies on the previous mode,
                hence flip indicates the total amount of weight updates so far.
            - **'update'**:
                The update size as the norm of the difference between the
                current weights and the previous weights for the current mode.
            - **'shrinkage'**: The shrinkage for the current mode.
            - **'objective'**:
                The mode discriminant objective value calculated as the sum of
                the eigenvalues of the generalized eigenvalue problem.
            - **'F_tr'** (extra): the overall trace-ratio discriminant objective value.
            - **'F_rt'** (extra): the overall ratio-trace discriminant objective value.

        Forward:
            - **'iteration'**,**'mode'** and **'flip'** as above.
            - **'update'**:
                The update size as the norm of the difference between the
                current activation patterns and the previous activation patterns
                for the current mode.
            - **'mse'** (extra): Overall reconstruction Mean Squared Error.
            - **'nmse'** (extra): Overall reconstruction Normalized Mean Squared Error.


        `train_info_['backward']` and `train_info_['forward']` can be used to
        initialize a pandas DataFrame.

    rank_: tuple
        A tuple with lenght equaling the tensor order containing the actual rank
        calculated using `theta` or set as `rank`.

    n_params_: int
        The total number of parameters in the backward model calculated as

            dim_1*rank_1 + dim_2*rank_2 + ... + dim_K*rank_K

        This is also the number of parameters in the forward model.

    References
    ----------
    [1] Phan, A. H., & Cichocki, A. (2010).
        Tensor decompositions for feature extraction and classification of high
        dimensional datasets. *Nonlinear Theory and Its Applications, IEICE*, 1(1), 37–68.

    [2] Van den Kerchove, A., Libert, A., Wittevrongel, B., & Van Hulle, M. M. (2022).
        Classification of event-related potentials with regularized spatiotemporal
        LCMV beamforming. *Applied Sciences*, 12(6), 2918.
    """

    def __init__(
        self,
        rank=None,
        theta=None,
        obj="tr",
        shrinkage="lw",
        toeplitz=None,
        solver="lanczos",
        max_iter=256,
        tol=1e-8,
        solver_params=None,
        verbose=False,
        extra_train_info=False,
        forward=False,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.rank = rank
        self.shrinkage = shrinkage
        self.toeplitz = toeplitz
        self.obj = obj
        self.solver = solver
        self.verbose = verbose
        self.solver_params = solver_params
        self.extra_train_info = extra_train_info
        self.theta = theta
        self.forward = forward

    def _validate(self, X, y=None):
        if not tl.is_tensor(X):
            raise ValueError("X must be a tensorly tensor object")
        if self.theta is None and self.rank is None:
            raise ValueError("Either theta or rank must be set.")
        if self.obj not in OBJECTIVES.keys():
            raise ValueError(f"objective must be one of {list(OBJECTIVES.keys())}")
        return X, y

    def fit(self, X, y, classes=None, class_counts=None):
        """Fit the estimator to the data.

        Fits the backward model. If `self.forward` is True, also fits the
        forward model.


        Parameters
        ----------
        X : tensorly.tensor of shape (n_samples, dim_1, dim_2, ..., dim_K)
            Training data.

        y : array-like of shape (n_samples), default=None
            Class labels.

        classes : list of obj
            List containing precomputed unique class labels in `y` to speed up
            GPU computation.

        class_counts : list of int
            Precomputed occurence counts of unique classes in `y` to speed up
            GPU computation.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        X, y = self._validate(X, y)
        # Calculate means, centering and classes once
        if classes is None or class_counts is None:
            self.classes_, class_counts = np.unique(y, return_counts=True)
            class_order = np.argsort(self.classes_)
            self.classes_ = self.classes_[class_order]
            class_counts = tl.tensor(class_counts[class_order])
        else:
            self.classes_ = classes
        self.classes_, class_counts = np.unique(y, return_counts=True)
        class_counts = tl.tensor(class_counts)
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
        """Fit the backward model to the data.

        Calculates `self.weights_`.
        Parameters
        ----------
        X : tensorly.tensor of shape (n_samples, dim_1, dim_2, ..., dim_K)
            Training data.

        y : array-like of shape (n_samples), default=None
            Class labels.

        X_centered : tensorly.tensor of shape (n_samples, dim_1, dim_2, ..., dim_K), default=None
            Precomputed centered input data obtained by subtracting the class
            means from the corresponding class samples to speed up computation.

        means: tensorly.tensor of shape (n_classes, dim_1, dim_2, ..., dim_k), default=None
            Precomputed class means to speed up computation.

        classes : list, default=None
            List of length n_classes containing precomputed unique class labels in `y` to speed up
            GPU computation.

        class_counts : list, default=None
            List of length n_classes containing precomputed occurence counts of unique classes in `y` to speed up
            GPU computation.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        X, y = self._validate(X, y)
        _, *shape = X.shape
        order = len(shape)

        # Determine classes and means and center data if not yet done in fit
        if classes is None or class_counts is None:
            self.classes_, class_counts = np.unique(y, return_counts=True)
            class_order = np.argsort(self.classes_)
            self.classes_ = self.classes_[class_order]
            class_counts = tl.tensor(class_counts[class_order])
        else:
            self.classes_ = classes
        if X_centered is None or means is None:
            self.means_, X_centered = center(X, y, self.classes_)
        else:
            self.means_ = means

        # Initialize train info
        self.train_info_ = dict(backward=[])

        # Initialize backward projections and covariances
        self._init_backward(X)

        # Calculate means and center
        means_centered = self.means_ - tl.mean(self.means_, axis=0)

        # Calculate total scatter
        scatter_t = self._calculate_scatter_t(X_centered, class_counts)

        # Iteratively find projections
        iterator = range(1, self.max_iter + 1)
        if self.verbose:
            iterator = tqdm(iterator, position=0, leave=True)
        for self._iter in iterator:
            converged = True
            for k in range(order):

                # Alternating partial projection
                modes = range(1, order + 1)
                X_centered_proj = tl.tenalg.multi_mode_dot(
                    X_centered,
                    self.weights_,
                    modes=modes,
                    skip=k,
                    transpose=True,
                )
                means_centered_proj = tl.tenalg.multi_mode_dot(
                    means_centered, self.weights_, modes=modes, skip=k, transpose=True
                )

                # Within-class and between-class scatter matrices
                scatter_w, scatter_b, shrinkage = self._calculate_scatter_wb(
                    X_centered_proj,
                    means_centered_proj,
                    k,
                    class_counts,
                )

                # Solve
                u, w = self._solve_backward_step(k, scatter_w, scatter_b, scatter_t[k])

                # Calculate update and check convergence
                if u.shape[-1] != self.weights_[k].shape[-1]:
                    update = np.inf
                else:
                    update = tl.norm(self.weights_[k] - u)
                    update /= tl.norm(self.weights_[k])
                converged = update < self.tol and converged

                # Assign new weights
                self.weights_[k] = u

                # Store mode training information
                self._store_backward_train_info(X, y, k, update, shrinkage, w)

            if self.verbose:
                iterator.set_description(f"Backward HODA model rank={self.rank_}")
            # Exit if converged in all modes
            if converged:
                break

        if not converged:
            warnings.warn(
                "Maximum number of iterations reached without convergence in backward fitting"
            )
        return self

    def _calculate_scatter_t(self, X_centered, class_counts):
        order = X_centered.ndim - 1
        scatter_t = [None] * order
        toeplitz = self.toeplitz
        if toeplitz is None:
            toeplit = tuple()
        for k in range(order):
            scatter_w, shrinkage = mode_scatter(
                X_centered,
                k,
                assume_centered=True,
                shrinkage=self.shrinkage,
                toeplitz=k in toeplitz,
            )
            scatter_b, _ = mode_scatter(
                self.means_, k, weights=class_counts, shrinkage=0
            )
            scatter_t[k] = scatter_w + scatter_b

        return scatter_t

    def _calculate_scatter_wb(
        self, X_centered_proj, means_centered_proj, k, class_counts
    ):
        order = X_centered_proj.ndim - 1
        modes = range(1, order + 1)
        if isinstance(self.shrinkage, tuple):
            shrinkage = self.shrinkage[k]
        else:
            shrinkage = self.shrinkage

        # Calculate within class scatter matrix
        toeplitz = self.toeplitz
        if toeplitz is None:
            toeplit = tuple()
        scatter_w, shrinkage = mode_scatter(
            X_centered_proj,
            k,
            assume_centered=True,
            shrinkage=shrinkage,
            toeplitz=k in toeplitz,
        )

        # Calculate between class scatter matrix
        scatter_b, _ = mode_scatter(
            means_centered_proj,
            k,
            weights=class_counts,
            shrinkage=0,
            assume_centered=True,
        )
        return scatter_w, scatter_b, shrinkage

    def _solve_backward_step(self, k, scatter_w, scatter_b, scatter_t):
        solver_params = self.solver_params
        if solver_params is None:
            solver_params = dict()

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
            u @ u.T @ scatter_t @ u @ u.T,
            rank=self.rank_[k],
            solver=self.solver,
            which="LA",
            init=tl.copy(self.weights_[k]),
            **solver_params,
        )

        if np.any(np.isnan(u)):
            raise LinAlgError("NaN in weights")

        return u, w

    def _store_backward_train_info(self, X, y, k, update, shrinkage, w):
        order = X.ndim - 1
        train_info_row = dict(
            iteration=self._iter,
            mode=k + 1,
            flip=(self._iter - 1) * order + k + 1,
            update=float(update),
            shrinkage=float(shrinkage),
            objective=float(tl.sum(tl.abs(w))),
        )
        if self.extra_train_info:
            Xt = self.transform(X)
            train_info_row.update(_backward_stats(Xt, y))
        self.train_info_["backward"].append(train_info_row)

    def fit_forward(self, X, y, X_centered=None, Xt=None):
        """Fit the forward model to the data.

        Calculates `self.activation_patterns_`.

        Parameters
        ----------
        X : tensorly.tensor of shape (n_samples, dim_1, dim_2, ..., dim_K)
            Training data.

        y : array-like of shape (n_samples), default=None
            Class labels.

        X_centered : tensorly.tensor of shape (n_samples, dim_1, dim_2, ..., dim_K), default=None
            Precomputed centered input data obtained by subtracting the class
            means from the corresponding class samples to speed up computation.

        Xt : tensorly.tensor of shape (n_samples, dim_1, dim_2, ..., dim_K), default=None
            Precomputed core tensor G to speed up computation.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = self._validate(X, y)
        _, *shape = X.shape
        order = len(shape)

        # Project
        if Xt is None:
            Xt = self.transform(X)

        # Initialize
        self._init_forward()
        self.train_info_["forward"] = []

        iterator = range(1, self.max_iter)
        if self.verbose:
            iterator = tqdm(iterator, position=0, leave=True)
            iterator.set_description("Forward model ")
        update = np.inf
        for i in iterator:
            converged = True
            for k in range(order):

                # Alternating partial projection
                modes = range(1, order + 1)
                G = tl.tenalg.multi_mode_dot(Xt, self.aps_, modes=modes, skip=k)

                ap, lambda_ = self._solve_forward_step(X, G, k)

                # Calculate update and convergence
                update = tl.norm(ap - self.aps_[k])
                update /= tl.norm(self.aps_[k])
                converged = update < self.tol and converged

                # Set new activation pattern
                self.aps_[k] = ap

                # Store training info
                self._store_forward_train_info(X, y, i, k, update, lambda_)
            if converged:
                break

        if not converged:
            warnings.warn(
                "Maximum number of iterations reached without convergence in forward fitting"
            )
        return self

    def _solve_forward_step(self, X, G, k, lambda_=0.0):
        # Least squares regression
        # modes = tuple([kk for kk in range(order + 1) if kk != k + 1])
        # XTX = tl.tenalg.tensordot(G, G, modes)
        # XTY = tl.tenalg.tensordot(G, X, modes)
        # ap = tl.solve(XTX, XTY).T

        # Ridge regression
        Xk = tl.unfold(X, k + 1)
        Gk = tl.unfold(G, k + 1)
        # TODO: regularization
        ap = ridge_regression(Gk.T, Xk.T, lambda_=lambda_).T

        if np.any(np.isnan(ap)):
            raise LinAlgError("NaN in aps")

        return ap, lambda_

    def _store_forward_train_info(self, X, y, i, k, update, lambda_):
        order = X.ndim - 1
        train_info_row = dict(
            iteration=i,
            mode=k + 1,
            flip=(i - 1) * order + k + 1,
            update=float(update),
            lambda_=float(lambda_),
        )
        if self.extra_train_info:
            Xt = self.transform(X)
            X_approx = self.inv_transform(Xt)
            train_info_row.update(_forward_stats(X, Xt, X_approx, y))
        self.train_info_["forward"].append(train_info_row)

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
        """Transform input data X to the core tensor G.

        Parameters
        ----------
        X : tensorly.tensor of shape (n_samples, dim_1, dim_2, ..., dim_K)
            Input data.

        y : ignored, default=None

        Returns
        -------
        Xt: tensorly.tensor of shape (n_samples, rank_1, rank_2, ..., rank_K)
            Core tensor G obtained as

                G = X ×₁ W₁ ×₂ W₂ × ... ×ₖ Wₖ
        """
        X, y = self._validate(X, y)
        order = len(X.shape) - 1
        Xt = tl.tenalg.multi_mode_dot(
            X, self.weights_, modes=range(1, order + 1), transpose=True
        )
        return Xt

    def inv_transform(self, Xt, y=None):
        """Reconstruct the original data from the core tensor G.

        Parameters
        ----------
        Xt: tensorly.tensor of shape (n_samples, rank_1, rank_2, ..., rank_K)
            Core tensor G.

        y : ignored, default=None

        Returns
        -------
        X : tensorly.tensor of shape (n_samples, dim_1, dim_2, ..., dim_K)
            Estimated reconstruction of the original input data obtained as

                X ≈ G ×₁ A₁ᵀ ×₂ A₂ᵀ × ... ×ₖ Aₖᵀ
        """
        Xt, y = self._validate(Xt, y)
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


def _backward_stats(Xt, y):
    n, *shape = Xt.shape
    p = math.prod(shape)
    stats = {
        "F_tr": float(f_multiway(Xt, y, method="tr")),
        "F_rt": float(f_multiway(Xt, y, method="rt")),
    }
    return stats


def _forward_stats(X, Xt, X_approx, y):
    mse = tl.metrics.regression.MSE(X, X_approx)
    nmse = mse / tl.metrics.regression.MSE(X, 0)
    stats = {
        "mse": float(mse),
        "nmse": float(nmse),
    }
    return stats


class BTTDA(BaseEstimator, TransformerMixin):
    """Block-Term Tensor Discriminant Analysis (BTTDA) tensor decomposition method.

    BTTDA extends HODA to extract multiple core tensors from the input data instead
    of only one, effectively decomposing the input data in several blocks. BTTDA
    works by fitting HODA models in an iterative deflation scheme.

        1. To obtain the first block, the HODA backward model is applied to the
           input data X to obtain the corresponding core tensor G_1.
        2. Next, the initial data is reconstructed as X_rec_1 using the forward
           model of the first block
        3. Now, the error between the initial data and the reconstructed data
           is computed as E_1 = X - X_rec_1.
        4. The core tensor of the second block G_2 is now computed from this error
           term E by again applying the HODA backward model.
        5. The input data (now the error E) is again reconstructed from G_2
           as E_rec_2.
        6. E_2 is then obtained as E_2 = E_1 - E_rec_2 and HODA is applied to E_2
        7. This continues for `n_blocks` iterations.

    These blocks extract additional discriminatory information further than the
    first HODA blocks and can be used for feature extraction.

    Parameters
    ----------
    ranks : iterable of length n_blocks,
        `ranks` determines the number of core blocks to extract and their rank.
        The length of `ranks` is the number of blocks. Elements can be the
        accepted values for the `rank` parameter of `HODA`. Must be set.

    hoda_params: dict
       Parameters to pass to the internal HODA model for the blocks. The `rank`
       parameter is overriden by the corresponding rank from the `ranks` parameter
       above.

    extra_train_info : bool, default=False
        If True, calculate and store additional statistics (e.g., objective values)
        during iterations. This slows down fitting.

    verbose : bool, default=False
        If True, print progress information during fitting.

    forward : bool, default=False
        If True, also fit the forward model for the last block.

    Attributes
    ---------
    classes_ : list of obj
        List of length n_classes unique classes occuring in `y`, in increasing order.

    blocks_ : list of bttda.hoda.HODA
        List of length n_blocks containing fitted HODA models representing
        the blocks.

    train_info_ : list of dict
        A dictionary storing statistics gathered during fitting.
        `train_info_` is a list of dictionaries for each block, storing
        key-value pairs for that iteration. Following keys are available,
        if `extra_train_info` is true, keys marked with 'extra' are calculated
        and stored.

        - **'block'**: The current block.
        - **'rank'**: The rank of the current block.
        - **'F_tr'** (extra): the overall trace-ratio discriminant objective value.
        - **'F_rt'** (extra): the overall ratio-trace discriminant objective value.
        - **'mse'** (extra): Overall reconstruction Mean Squared Error.
        - **'nmse'** (extra): Overall reconstruction Normalized Mean Squared Error.

        `train_info_` can be used to initialize a pandas DataFrame.

    n_blocks_: int
        The number of extracted blocks. `n_blocks_` can be smaller than the length
        of `ranks` if errors occured during block fitting.

    n_params_:
    The total number of parameters in the backward model calculated as

        block_1.n_params_ + block_2.n_params_ + ... + block_K.n_params_

    This is also the number of parameters in the forward model.


    Notes
    -----
    If `len(ranks)` is 1, only one block is extracted. In this special case, the
    BTTDA model and the HODA model are equivalent.

    If each element of ranks is 1 or (1, 1, ..., 1), a series of rank 1 blocks
    is extracted. In this special case, the BTTDA model is called PARAFACDA after
    the rank-1 PARAFAC/CPD tensor decomposition.
    """

    def __init__(
        self,
        ranks=None,
        hoda_params=None,
        extra_train_info=False,
        verbose=False,
        forward=True,
    ):
        self.ranks = ranks
        self.hoda_params = hoda_params
        self.extra_train_info = extra_train_info
        self.verbose = verbose
        self.forward = forward

    def _validate(self, X, y=None):
        if not tl.is_tensor(X):
            raise ValueError("X must be a tensorly tensor object")
        return X, y

    def fit(self, X, y=None, blocks=None):
        """Fit the estimator to the data.

        Parameters
        ----------
        X : tensorly.tensor of shape (n_samples, dim_1, dim_2, ..., dim_K)
            Training data.

        y : array-like of shape (n_samples), default=None
            Class labels.

        blocks : list of HODA, default=None
            Precomputed list of fitted HODA blocks. If `blocks` is not None set,
            the `ranks` parameter is ignored for the first set of blocks corresponding
            to the length of `blocks` and the model is instead specified
            by these blocks instead of fitting new ones.
            If the requested number of blocks is greater than the length of `blocks`,
            further blocks will be fitted given these earlier blocks.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = self._validate(X, y)
        n_samples, *shape = X.shape

        self.classes_, class_counts = np.unique(y, return_counts=True)
        class_order = np.argsort(self.classes_)
        self.classes_ = self.classes_[class_order]
        class_counts = tl.tensor(class_counts[class_order])

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

                # Store train info
                self._store_train_info(X, y, block)

            except (LinAlgError, ValueError) as e:
                warnings.warn(RuntimeWarning(str(e)))
                break

        return self

    def _store_train_info(self, X, y, block):
        train_info_row = dict()
        train_info_row["block"] = self.n_blocks_
        train_info_row["rank"] = block.rank_

        if self.extra_train_info:
            Xt = self.transform(X)
            X_approx = self.inv_transform(Xt)
            train_info_row.update(_backward_stats(Xt, y))
            train_info_row.update(_forward_stats(X, Xt, X_approx, y))
        self.train_info_.append(train_info_row)

    @property
    def n_blocks_(self):
        return len(self.blocks_)

    @property
    def n_params_(self):
        return sum([b.n_params_ for b in self.blocks_])

    def transform(
        self,
        X,
        y=None,
        blocks=None,
        n_blocks=None,
        return_err=False,
        flatten=True,
        **kwargs,
    ):
        """Transform input data X to the core tensor G.

        Parameters
        ----------
        X : tensorly.tensor of shape (n_samples, dim_1, dim_2, ..., dim_K)
            Input data.

        y : ignored, default=None

        blocks : None or iterable of HODA, default=None
            If `blocks`is not None, use this list of HODA blocks instead of
            the fitted blocks to perform the transformatoin.

        n_blocks : None or int, default=None
            Only use the first `n_blocks` blocks of `self.blocks_` or `blocks`
            to perform the transformation. If None, use all blocks.

        return_err : bool, default is False
            If True, return the residual error after transforming the input
            data `X` to the block core tensors.

        flatten: bool, default is True
            If False, return the transformed block core tensors as a list of
            tensors with potentially different dimensions. If True, return a
            flattened output samples as a single array, losing the core tensor
            structure. This is useful if BTTDA is used for feature extraction.


        Returns
        -------
        Xt: list of tensorly.tensor of shape (n_samples, rank_1b, rank_2b, ..., rank_Kb) or tensorly.tensoror of shape (n_samples, flattened_ranks)
            The core tensors of each block after obtained by transforming th
            residual error in the deflation scheme. The output is either represented
            as a list of tensors with different shapes like

                [(n_samples, b1_rank_1, ..., b1_rank_K), ..., (n_samples, bB_rank_1, ..., bB_rank_K)]

             if `flatten` is False, or a single array of size (n_samples, flattened_ranks) like

                (n_samples, sum(block_1.rank_) + ... + sum(block_K.rank_))

            if `flatten` is True.

        err : tensorly.tensor of shape (n_samples, dim_1, dim_2, ..., dim_K)
            The residual error left after extracting each block core tensor
            in the deflation scheme. Only returned if `return_err` is True.
        """
        X, y = self._validate(X, y)
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

        if flatten:
            Gs = [G.reshape((n_samples, -1)) for G in Gs]
            Gs = tl.concatenate(Gs, axis=1)
        if return_err:
            return Gs, err
        return Gs

    def inv_transform(self, Xt, y=None, n_blocks=None):
        """Reconstruct the original data from the flattened block core tensors.

        Parameters
        ----------
        Xt: tensorly.tensor of shape (n_samples, flattened_ranks)
            Flattened transformed block core tensors.

        y : ignored, default=None

        n_blocks : int or None, default=None
            If n_blocks is not None, only use the first `n_blocks` blocks of the
            transformed data to reconstruct the original data.

        Returns
        -------
        X : tensorly.tensor of shape (n_samples, dim_1, dim_2, ..., dim_K)
            Estimated reconstruction of the original input data.
        """
        X, y = self._validate(Xt, y)
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
    y,
    classes=None,
    class_counts=None,
    assume_centered=False,
    means=None,
    method="tr",
):
    """f_multiway.
    Calculate MANOVA-like multivariate discriminability F-statistic.

    Parameters
    ----------
    X : tensorly.tensor of shape (n_samples, *shape)
        Input data.

    y : array-like of obj
        Class labels of length `n_samples`.

    classes : list of obj or None, default is None
        List containing precomputed unique class labels in `y` to speed up
        GPU computation. If None, compute the classes now.

    class_counts : list of int or None, default is None
        Precomputed occurence counts of unique classes in `y` to speed up
        GPU computation. If None, compute the class counts now.

    assume_centered : bool, default=False
        If True, do not center the data by subtracting class means, instead assume
        this is already done. In this case, `means` must also be set.

    means : tensorly.tensor of shape (n_classes, *shape) or None, default=None
        Precomputed class means.

    method : {'tr', 'rt'}, default='tr'
        Determine the multivariate statistic of discriminability as the
        Trace-Ratio criterion ('tr') or the Ratio-Trace critarion ('rt').
        Ratio-Trace requires solving a generalized eigenvalue problem, while
        Trace-Ratio can be computed analytically.

    Returns
    -------
    F : float
        The multivariate statistic of discriminability, either computed as the
        trace-ratio of the input data or the ratio-trace.
    """
    n_samples, *shape = X.shape
    if not tl.is_tensor(X):
        X = tl.tensor(X)
    if classes is None or class_counts is None:
        classes, class_counts = np.unique(y, return_counts=True)
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
            solver="lobpcg",
            eigvals_only=True,
        )
        F = tl.sum(w)
    else:
        raise ValueError(
            "method must be either 'tr' (trace-ratio) or 'rt' (ratio-trace)"
        )
    return F


def f_oneway(X, y, classes=None, class_counts=None):
    """Univariate ANOVA discriminability F-statistic.

     ANOVA discriminability statistic as in `scipy.stats.f_oneway` or
     `sklearn.feature_selection.f_classif`, but implemented for `tensorly`.

     Parameters
     ----------
    X : tensorly.tensor of shape (n_samples, *shape)
         Input data.

     y : array-like of obj
         Class labels of length `n_samples`.

     classes : list of obj or None, default is None
         List containing precomputed unique class labels in `y` to speed up
         GPU computation. If None, compute the classes now.

     class_counts : list of int or None, default is None
         Precomputed occurence counts of unique classes in `y` to speed up
         GPU computation. If None, compute the class counts now.

     Returns
     -------
     F : tensorly.tensor of shape (*shape)
         Univariate ANOVA F-statistic for each feature.
    """
    n_samples, *shape = X.shape
    order = len(shape)
    if classes is None or class_counts is None:
        classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    ss_alldata = tl.sum(X**2, axis=0)
    sums_per_class, _ = center(X, y, classes)
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
