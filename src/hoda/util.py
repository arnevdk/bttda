import warnings

import numpy as np
import tensorly as tl
from numpy.linalg import LinAlgError

from hoda.backend import fdtrc, lanczos, lobpcg
from hoda.cov import mode_scatter

try:
    import cupy
except ImportError:
    pass


def norm_fro(A):
    return tl.sqrt(tl.sum(A**2))


def trunc_eigh(
    A,
    B=None,
    rank=None,
    largest=True,
    method="lanczos",
    solver_params=None,
):
    """

    SVD eigensolver can only be used if  B^-1@A is semi-positive definite
    """
    solver_params = solver_params or dict()
    if method == "lanczos":
        try:
            v, w = lanczos(A, B=B, rank=rank, largest=largest, **solver_params)
            if np.any(np.isnan(v)):
                raise LinAlgError
        except LinAlgError as e:
            warnings.warn(f"lanczos failed with error {e}")
            v, w = lanczos(
                A, B=B, rank=rank, largest=largest, force_spd=True, **solver_params
            )

    elif method == "svd":
        solver_params.setdefault("method", "truncated_svd")
        if largest:
            solver_params["n_eigenvecs"] = rank
        else:
            solver_params["n_eigenvecs"] = None
        if B is None:
            v, w, _ = tl.tenalg.svd_interface(A, **solver_params)
        else:
            v, w, _ = tl.tenalg.svd_interface(tl.solve(B, A), **solver_params)
        if not largest:
            w = w[-rank:]
            v = v[:, -rank:]

    elif method == "lobpcg":
        init = solver_params.pop("init", tl.eye(A.shape[0]))
        init = init[:, :rank]
        try:
            w, v = lobpcg(A, init, B=B, largest=largest, **solver_params)
        except (AttributeError, LinAlgError, Exception, ValueError) as e:
            warnings.warn(
                f"lobpcg failed with error {e}, falling back to lanczos solver with SPD constraint"
            )
            v, w = lanczos(
                A, B=B, rank=rank, largest=largest, force_spd=True, **solver_params
            )
    else:
        raise ValueError("Solver must be one of ['lanczos', 'lobpcg', 'svd']")
    # Flip sign
    sign = tl.sign(v[0, :])
    v *= sign
    # Normalize
    v = v / tl.norm(v)
    # sort
    idc = np.argsort(w)
    w = w[idc]
    v = v[:, idc]
    return v, w


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
        F = tr_scatter_b / tr_scatter_w
    elif method == "rt":
        X_centered_flat = tl.unfold(X_centered, 0)

        scatter_w, _ = mode_scatter(
            X_centered_flat, 0, assume_centered=True, shrinkage="lw"
        )
        means_flat = tl.unfold(means, 0)
        scatter_b, _ = mode_scatter(means_flat, 0, assume_centered=False, shrinkage=0)
        _, w = trunc_eigh(scatter_b, scatter_w, rank=None, method="lanczos")
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


def combine_pvalues(pvalues, axis=None, method="fisher", weights=None):
    """
    Methods for combining the p-values of independent tests bearing upon the
    same hypothesis.

    Parameters
    ----------
    p: array_like, 1-D
        Array of p-values assumed to come from independent tests.
    method: str
        Name of method to use to combine p-values. The following methods are
        available:
        - "fisher": Fisher's method (Fisher's combined probability test)
        - "stouffer": Stouffer's Z-score method
    weights: array_like, 1-D, optional
        Optional array of weights used only for Stouffer's Z-score method.

    Returns
    -------
    statistic: float
        The statistic calculated by the specified method:
        - "fisher": The chi-squared statistic
        - "stouffer": The Z-score
    pval: float
        The combined p-value.

    Notes
    -----
    Fisher's method (also known as Fisher's combined probability test) [1]_ uses
    a chi-squared statistic to compute a combined p-value. The closely related
    Stouffer's Z-score method [2]_ uses Z-scores rather than p-values. The
    advantage of Stouffer's method is that it is straightforward to introduce
    weights, which can make Stouffer's method more powerful than Fisher's
    method when the p-values are from studies of different size [3]_ [4]_.

    Fisher's method may be extended to combine p-values from dependent tests
    [5]_. Extensions such as Brown's method and Kost's method are not currently
    implemented.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Fisher%27s_method
    .. [2] http://en.wikipedia.org/wiki/Fisher's_method#Relation_to_Stouffer.27s_Z-score_method
    .. [3] Whitlock, M. C. "Combining probability from independent tests: the
           weighted Z-method is superior to Fisher's approach." Journal of
           Evolutionary Biology 18, no. 5 (2005): 1368-1373.
    .. [4] Zaykin, Dmitri V. "Optimally weighted Z-test is a powerful method
           for combining probabilities in meta-analysis." Journal of
           Evolutionary Biology 24, no. 8 (2011): 1836-1841.
    .. [5] https://en.wikipedia.org/wiki/Extensions_of_Fisher%27s_method

    """
    if method == "fisher":
        shape = pvalues.shape
        order = len(shape)
        statistic = -2 * tl.sum(tl.log(pvalues), axis=axis)
        k = math.prod([shape[a] for a in range(order) if a in axis])
        df = 2 * k
        pval = chdtrc(df, statistic)
        return (statistic, pval)

    elif method == "edgington" or method == "average":
        p = tl.mean(pvalues, axis=axis)
        return p, p

    #    elif method == 'pearson':
    #        statistic = 2 * np.sum(np.log1p(-pvalues))
    #        pval = distributions.chi2.cdf(-statistic, 2 * len(pvalues))
    #    elif method == 'mudholkar_george':
    #        normalizing_factor = np.sqrt(3/len(pvalues))/np.pi
    #        statistic = -np.sum(np.log(pvalues)) + np.sum(np.log1p(-pvalues))
    #        nu = 5 * len(pvalues) + 4
    #        approx_factor = np.sqrt(nu / (nu - 2))
    #        pval = distributions.t.sf(statistic * normalizing_factor
    #                                  * approx_factor, nu)
    #    elif method == 'tippett':
    #        statistic = np.min(pvalues)
    #        pval = distributions.beta.cdf(statistic, 1, len(pvalues))
    #    elif method == 'stouffer':
    #        if weights is None:
    #            weights = np.ones_like(pvalues)
    #        elif len(weights) != len(pvalues):
    #            raise ValueError("pvalues and weights must be of the same size.")
    #
    #        weights = np.asarray(weights)
    #        if weights.ndim != 1:
    #            raise ValueError("weights is not 1-D")
    #
    #        Zi = distributions.norm.isf(pvalues)
    #        statistic = np.dot(weights, Zi) / np.linalg.norm(weights)
    #        pval = distributions.norm.sf(statistic)
    #
    else:
        raise ValueError(
            f"Invalid method {method!r}. Valid methods are 'fisher', "
            "'pearson', 'mudholkar_george', 'tippett', and 'stouffer'"
        )
