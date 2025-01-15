import math

import numpy as np
import scipy.linalg
import tensorly as tl
from numpy.linalg import LinAlgError

from hoda.cov import mode_scatter

try:
    import cupy
except ImportError:
    pass


def norm_fro(A):
    return tl.sqrt(tl.sum(A**2))


def solve_gevdh(
    A, B=None, solver="lanczos", rank=None, eigvals_only=False, **solver_params
):
    if solver == "lanczos":
        subset = None
        n = A.shape[-1]
        if rank is not None:
            subset = [n - rank, n - 1]
        res = scipy.linalg.eigh(
            A,
            b=B,
            check_finite=False,
            subset_by_index=subset,
            eigvals_only=eigvals_only,
            **solver_params
        )
        if isinstance(res, tuple):
            res = (res[1], res[0])
    if solver == "svd":
        raise NotImplementedError
    if solver == "lobpcg":
        raise NotImplementedError
    return res


def center(X, y, classes=None):
    _, *shape = X.shape
    order = len(shape)
    if classes is None:
        classes = np.unique(y)
    n_classes = len(classes)

    means = tl.zeros((n_classes, *shape))
    if tl.get_backend() == "cupy":
        X_centered = tl.zeros((n_classes, *X.shape))
        full_nan = cupy.full_like(X, cupy.nan)
        for ci, c in enumerate(classes):
            where = y == c
            where = cupy.array(where)
            where = np.expand_dims(where, axis=tuple(np.arange(1, order + 1)))
            X_where = cupy.where(
                where,
                X,
                full_nan,
            )
            means[ci] = cupy.nanmean(X_where, axis=0)
            X_centered[ci] = X_where - means[ci]
        X_centered = cupy.nansum(X_centered, axis=0)
    else:
        X_centered = []
        for ci, c in enumerate(classes):
            X_where = X[y == c]
            means[ci] = tl.mean(X_where, axis=0)
            X_centered.append(X_where - means[ci])
        X_centered = tl.concatenate(X_centered, axis=0)
    return means, X_centered


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
        F = (tr_scatter_b / tr_scatter_w) * ((n_classes - 1) / (n_samples - n_classes))
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


def r_squared(
    X,
    y=None,
    classes=None,
    class_counts=None,
    assume_centered=False,
    means=None,
):
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
    SS_res = norm_fro(X_centered) ** 2
    SS_tot = norm_fro(X) ** 2
    return 1 - SS_res / SS_tot


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


def lstsq_ridge(X, y, lambda_=1):
    I = tl.eye(X.shape[1])
    XTX = X.T @ X
    n_features = XTX.shape[0]
    pseudo_inverse = pinv(XTX + lambda_ * tl.trace(XTX) * I / n_features) @ X.T
    return pseudo_inverse @ y
