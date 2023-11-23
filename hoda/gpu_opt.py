import warnings

import ipdb
import numpy as np
import tensorly as tl

try:
    import cupy
except ImportError:
    pass


from hoda.tenalg import chdtrc


def center(X, y, classes=None):
    _, *shape = X.shape
    order = len(shape)
    if classes is None:
        classes = np.unique(y)
    n_classes = len(classes)

    means = tl.zeros((n_classes, *shape), dtype=X.dtype)
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


def ledoit_wolf_shrinkage(X, assume_centered=False, block_size=1000):
    """Estimate the shrunk Ledoit-Wolf covariance matrix.
    Read more in the :ref:`User Guide <shrunk_covariance>`.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data from which to compute the Ledoit-Wolf shrunk covariance shrinkage.
    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, data will be centered before computation.
    block_size : int, default=1000
        Size of blocks into which the covariance matrix will be split.
    Returns
    -------
    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.
    Notes
    -----
    The regularized (shrunk) covariance is:
    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)
    where mu = trace(cov) / n_features
    """
    # for only one feature, the result is the same whatever the shrinkage
    if len(X.shape) == 2 and X.shape[1] == 1:
        return 0.0
    if X.ndim == 1:
        X = tl.reshape(X, (1, -1))

    if X.shape[0] == 1:
        warnings.warn(
            "Only one sample available. You may want to reshape your data array"
        )
    n_samples, n_features = X.shape

    # optionally center data
    if not assume_centered:
        X = X - X.mean(0)

    # A non-blocked version of the computation is present in the tests
    # in tests/test_covariance.py

    # number of blocks to split the covariance matrix into
    n_splits = int(n_features / block_size)
    X2 = X**2
    emp_cov_trace = tl.sum(X2, axis=0) / n_samples
    mu = tl.sum(emp_cov_trace) / n_features
    beta_ = 0.0  # sum of the coefficients of <X2.T, X2>
    delta_ = 0.0  # sum of the *squared* coefficients of <X.T, X>
    # starting block computation
    for i in range(n_splits):
        for j in range(n_splits):
            rows = slice(block_size * i, block_size * (i + 1))
            cols = slice(block_size * j, block_size * (j + 1))
            beta_ += tl.sum(tl.dot(X2.T[rows], X2[:, cols]))
            delta_ += tl.sum(tl.dot(X.T[rows], X[:, cols]) ** 2)
        rows = slice(block_size * i, block_size * (i + 1))
        beta_ += tl.sum(tl.dot(X2.T[rows], X2[:, block_size * n_splits :]))
        delta_ += tl.sum(tl.dot(X.T[rows], X[:, block_size * n_splits :]) ** 2)
    for j in range(n_splits):
        cols = slice(block_size * j, block_size * (j + 1))
        beta_ += tl.sum(tl.dot(X2.T[block_size * n_splits :], X2[:, cols]))
        delta_ += tl.sum(tl.dot(X.T[block_size * n_splits :], X[:, cols]) ** 2)
    delta_ += tl.sum(
        tl.dot(X.T[block_size * n_splits :], X[:, block_size * n_splits :]) ** 2
    )
    delta_ /= n_samples**2
    beta_ += tl.sum(
        tl.dot(X2.T[block_size * n_splits :], X2[:, block_size * n_splits :])
    )
    # use delta_ to compute beta
    beta = 1.0 / (n_features * n_samples) * (beta_ / n_samples - delta_)
    # delta is the sum of the squared coefficients of (<X.T,X> - mu*Id) / p
    delta = delta_ - 2.0 * mu * emp_cov_trace.sum() + n_features * mu**2
    delta /= n_features
    # get final beta as the min between beta and delta
    # We do this to prevent shrinking more than "1", which would invert
    # the value of covariances
    beta = tl.min(tl.tensor([beta, delta]))
    # finally get shrinkage
    # shrinkage = 0 if beta == 0 else beta / delta
    shrinkage = beta / delta
    return shrinkage


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
        statistic = -2 * tl.sum(tl.log(pvalues), axis=axis)
        df = 2 * tl.prod(tl.tensor(pvalues.shape)[axis])
        pval = chdtrc(df, statistic)
        return (statistic, pval)
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
