import math

import tensorly as tl


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
        pval = backend.scipy.special.chdtrc(df, statistic)
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
