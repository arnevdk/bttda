import math

import tensorly as tl
from sklearn.base import BaseEstimator

from bttda.util import get_eye, toeplitz


def mode_scatter(
    X, k, weights=None, shrinkage=0, toeplitz=False, assume_centered=False
):
    """Calculate the scatter matrix of the unfolding of a tensor along a given mode.

    Parameters
    ----------
    X : tensorly.tensor of shape (n_samples, dim_1, ..., dim_k, ..., dim_K)
        The input data.
    k : int
        The mode along which to compute the scatter matrix.
    weights : None or array-like of shape (n_samples)
        Optional sample weights for weighted scatter matrix.

    shrinkage : {'lw', 'oas', 'ss', 'ell', 'loocv'} default='lw'
        Shrinkage method or factor used to regularize the within-class scatter
        matrix.

        - If a **float** between 0.0 and 1.0, the within-class scatter is directly
          regularized as::

              S_shrunk = (1 - shrinkage) * S + shrinkage * mean(diag(S)) * I

        - If a **string**, the shrinkage factor is estimated automatically using
          the specified method:

              - **'lw'** : Ledoit–Wolf shrinkage [1].
              - **'oas'** : Oracle Approximating Shrinkage [2].
              - **'ss'** : Schäfer–Strimmer shrinkage [3].
              - **'ell'** : Robust shrinkage for elliptical distributions [4].
              - **'loocv'** : Closed-form Leave-One-Out Cross-Validation shrinkage [5].


    toeplitz : bool, default=False
        If True, impose a Toeplit matrix structure on the scatter matrix.

    assume_centered : bool, default=False
        If True, do not center the data by subtracting the mean, instead assume
        this is already done.



    References
    ----------
    [1] Ledoit, O., & Wolf, M. (2003). Honey, I shrunk the sample covariance
        matrix. Ledoit, Olivier, and Michael Wolf. "Honey, I shrunk the sample
        covariance matrix." (2003).
    [2] Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage
        algorithms for MMSE covariance estimation. IEEE transactions on signal
        processing, 58(10), 5016-5029.
    [3] Schäfer, J., & Strimmer, K. (2005). A shrinkage approach to large-scale
        covariance matrix estimation and implications for functional genomics.
        Statistical applications in genetics and molecular biology, 4(1).
    [4] Chen, Y., Wiesel, A., & Hero, A. O. (2011). Robust shrinkage estimation
        of high-dimensional covariance matrices. IEEE Transactions on Signal
        Processing, 59(9), 4097-4107.
    [5] Tong, J., Hu, R., Xi, J., Xiao, Z., Guo, Q., & Yu, Y. (2018). Linear
        shrinkage estimation of covariance matrices using low-complexity
        cross-validation. Signal Processing, 148, 223-233.
    """

    n_samples, *shape = X.shape
    order = len(shape)
    n_features = shape[k]
    # Determine mode scatter
    modes = [0] + [kk + 1 for kk in range(order) if kk != k]
    if not assume_centered:
        X = X - tl.mean(X, axis=0)

    if weights is None:
        weights = tl.ones(n_samples)
    weights = tl.reshape(
        weights, (n_samples,) + (1,) * (X.ndim - 1)
    )  # Expands to match X

    scatter = tl.tenalg.tensordot(X * weights, X, (modes, modes))
    if toeplitz:
        scatter = force_toeplitz(scatter)

    # Determine shrinkage
    if shrinkage == "lw":
        if scatter.shape[0] == 1:
            shrinkage = 0
        else:
            Xf = tl.unfold(X, k + 1)
            shrinkage = ledoit_wolf_shrinkage(
                Xf.T,
                scatter / (Xf.shape[1] - 1),
                assume_centered=assume_centered,
            )
    elif shrinkage == "oas":
        n = n_samples * math.prod(shape) / shape[k]
        cov = scatter / (n - 1)
        shrinkage = oas(cov, n)
    elif shrinkage == "ss":
        raise NotImplementedError
    elif shrinkage == "ell":
        raise NotImplementedError
    elif shrinkage == "loocv":
        raise NotImplementedError
    elif isinstance(shrinkage, str):
        raise ValueError(
            "shrinkage should be either float or one of ['lw', 'oas', 'ss', 'ell', 'loocv']"
        )

    trace = tl.trace(scatter)
    scale = trace / n_features
    target = get_eye(n_features) * scale
    scatter = (1 - shrinkage) * scatter + shrinkage * target

    return scatter, shrinkage


def force_toeplitz(A):
    """Find the nearest Toeplitz matrix.

    Calculated by setting each subdiagonal to its mean value.

    Parameters
    ----------
    A : tensorly.tensor of shape (n,n)
        Input matrix A.

    Returns
    -------
    toep : tensorly.tensor of shape (n,n)
        The nearest Toeplitz matrix to `A`.
    """
    n, _ = A.shape
    toep = tl.zeros(n)
    for i in range(n):
        diag = tl.diag(A, k=i)
        toep[i] = tl.mean(diag)
    return toeplitz(toep)


def ledoit_wolf_shrinkage(X, emp_cov, assume_centered=True):
    """Calculate the Ledoit-Wolf shrinkage factor.

    Efficient, GPU cabable `tensorly` implementation of
    `sklearn.covariance.ledoit_wolf_shrinkage`.

    Parameters
    ----------
    X : tensorly.tensor of shape (n_samples, n)
        Input data

    emp_cov : tensorly.tensor of shape (n,n)
        Empirical covariance matrix.
        emp_cov

    assume_centered : bool, default=False
        If True, do not center the data by subtracting the mean, instead assume
        this is already done.

    References
    ----------
    [1] Ledoit, O., & Wolf, M. (2003). Honey, I shrunk the sample covariance
        matrix. Ledoit, Olivier, and Michael Wolf. "Honey, I shrunk the sample
        covariance matrix." (2003).

    """
    n_samples, n_features = X.shape
    if n_features == 1:
        return 0
    if not assume_centered:
        X = X - X.mean(0)

    mu = tl.trace(emp_cov) / n_features
    delta_ = emp_cov.copy()
    shape = delta_.shape
    delta_flat = tl.base.tensor_to_vec(delta_)
    delta_flat[:: n_features + 1] -= mu
    delta_ = tl.base.vec_to_tensor(delta_, shape)
    delta = tl.sum(delta_**2) / n_features
    X2 = X**2
    beta_ = (
        1.0
        / (n_features * n_samples)
        * tl.sum(tl.dot(X2.T, X2) / n_samples - emp_cov**2)
    )
    beta_delta = tl.stack([beta_, delta])
    beta = tl.min(beta_delta)
    shrinkage = beta / delta
    shrinkage = tl.clip(shrinkage, a_min=0.0, a_max=1.0)
    return shrinkage


def oas(emp_cov, n_samples):
    """Calculate the Oracle Approximating Shrinkage factor.

    Efficient, GPU cabable `tensorly` implementation of
    `sklearn.covariance.oas`.

    Parameters
    ----------
    emp_cov : tensorly.tensor of shape (n,n)
        Empirical covariance matrix.
        emp_cov

    n_samples : int
        Number of samples in the input data.


    References
    ----------
    [1] Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage
        algorithms for MMSE covariance estimation. IEEE transactions on signal
        processing, 58(10), 5016-5029.

    """
    n_features = emp_cov.shape[0]
    if n_features == 1:
        return 0

    p = n_features
    n = n_samples
    S = emp_cov

    num = (1 - 2 / p) * tl.trace(S**2) + tl.trace(S) ** 2
    den = (n + 1 - 2 / p) * (tl.trace(S**2) - tl.trace(S) ** 2 / n_features)
    shrinkage = num / den
    shrinkage = min(shrinkage, 1)
    shrinkage = max(shrinkage, 0)
    return shrinkage
