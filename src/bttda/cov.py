import math

import tensorly as tl
from sklearn.base import BaseEstimator

from bttda.util import get_eye, toeplitz


def mode_scatter(
    X, k, weights=None, shrinkage=0, toeplitz=None, taper=False, assume_centered=False
):
    """Calculate the scatter matrix along a given tensor mode."""

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
    if toeplitz is not None and k in toeplitz:
        scatter = force_toeplitz(scatter, taper=taper)

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
        shrinkage = schaefer_strimmer_shrinkage(X, k)
    elif shrinkage == "ell1":
        raise NotImplementedError
    elif shrinkage == "ell2":
        raise NotImplementedError
    elif shrinkage == "ell3":
        raise NotImplementedError
    elif shrinkage == "loocv":
        raise NotImplementedError
    else:
        raise ValueError(
            "shrinkage should be either float or one of ['lw', 'oas', 'ss', 'ell1', 'ell2', 'ell3', 'loocv']"
        )

    trace = tl.trace(scatter)
    scale = trace / n_features
    target = get_eye(n_features) * scale
    scatter = (1 - shrinkage) * scatter + shrinkage * target

    return scatter, shrinkage


def force_toeplitz(A, taper=False):
    n, _ = A.shape
    toep = tl.zeros(n)
    for i in range(n):
        diag = tl.diag(A, k=i)
        toep[i] = tl.mean(diag)

    if taper:
        taper = tl.arange(len(toep), 0, -1) - 1
        toep = toep * taper
    return toeplitz(toep)


def ledoit_wolf_shrinkage(X, emp_cov, assume_centered=True):
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
    """Estimate covariance with the Oracle Approximating Shrinkage algorithm.

    The formulation is based on [1]_.
    [1] "Shrinkage algorithms for MMSE covariance estimation.",
        Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O.
        IEEE Transactions on Signal Processing, 58(10), 5016-5029, 2010.
        https://arxiv.org/pdf/0907.4698.pdf
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
