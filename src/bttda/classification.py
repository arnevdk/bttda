import math
import warnings

import numpy as np
import pandas as pd
import tensorly as tl
from joblib import Parallel, delayed
from sklearn.base import (BaseEstimator, ClassifierMixin, TransformerMixin,
                          clone)
from sklearn.feature_selection import SelectFdr, f_classif
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold

from bttda.hoda import BTTDA


class ZScore(BaseEstimator, TransformerMixin):
    """Standardize data based on fitted mean and standard deviation.

    Multi-dimensional implementation of a standardization
    sklear preprocessor.

    Attributes
    ----------
    mean_ : tensorly.tensor of shape (*shape)
        The mean of the training data.

    std_  : tensorly.tensor of shape (*shape)
        The standard deviation of the training data.
    """

    def fit(self, X, y=None):
        """Fit the preprocessor to the data.

        Parameters
        ----------
        X : tensorly.tensor of shape (n_samples, *shape)
            Input training data.
        y : ignored, default=None

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.mean_ = tl.mean(X, axis=(0, -1))
        self.std_ = np.std(X, axis=(0, -1))
        return self

    def transform(self, X, y=None):
        """Standardize using fitted mean and standard deviation.

        Parameters
        ----------
        X : tensorly.tensor of shape (n_samples, *shape)
            Input data to standardize
        y : ignored, default=None

        Returns
        -------
        Xt : tensorly.tensor of shape (n_samples, *shape)
            Standardized data.
        """
        shape = (1, *(X.shape[1:-1]), 1)
        mean = self.mean_.reshape(shape)
        std = self.std_.reshape(shape)
        return (X - mean) / std

    def inv_transform(self, X, y=None):
        """Inverse of the standardization using the fitted mean and standard deviation.

        Parameters
        ----------
        X : tensorly.tensor of shape (n_samples, *shape)
            Standardized input data
        y : ignored, default=False

        Returns
        -------
        Xt : tensorly.tensor of shape (n_samples, *shape)
            Scaled data after inverse of the standardization.
        """
        shape = (1, *(X.shape[1:-1]), 1)
        mean = self.mean_.reshape(shape)
        std = self.std_.reshape(shape)
        return (X * std) + mean


class SelectFCutoff(BaseEstimator, TransformerMixin):
    """Feature selector based on univariate F-statistic and a threshold.

    Parameters
    ----------
    cutoff : float
        F-statistic threshold > 0 below which to reject features.

    Attributes
    ----------
    scores_ : array-like of float of shape (n_features)
        The univariate F-statistic scores associated with each feature.

    support_ : array-like of bool of shape (n_features)
        True when the corresponding feature should be kept and False if it
        should be dropped.
    """

    def __init__(self, cutoff=1):
        self.cutoff = cutoff

    def fit(self, X, y=None):
        """Fit the feature selector to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input training data.
        y : array-like of obj shape (n_samples)
            Class labels.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.scores_, self.p_values_ = f_classif(X, y)
        self.scores_[np.isinf(self.scores_)] = 0
        self.scores_[np.isnan(self.scores_)] = 0
        self.support_ = self.scores_ > self.cutoff
        if not np.any(self.support_):
            self.support_ = np.zeros(len(self.support_), dtype=bool)
            self.support_[np.argmax(self.scores_)] = True
        return self

    def transform(self, X, y=None):
        """Drop features with F-statistic < `cutoff`

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.
        y : ignored, default=False

        Returns
        -------
        X : array-like of shape (n_samples, n_selected_features)
            Data retaining with only selected features.
        """
        return X[:, self.support_]


class BTTDACV(BTTDA):
    """BTTDA with cross-validated hyperparameter selection.

    To apply BTTDA effectively, HODA hyperparameter `theta` determining the rank
    of the blocks and `n_blocks`, the number of blocks, must be set to reasonable
    values. These can be obtained by performing a cross-validated grid search
    on a relevant classification problem.

    Given a classifier model to be applied after BTTDA feature extraction,
    different hyperparameter candidates are tested and the set of candidates
    with the highest cross-validated score is selected.

    For each cross-validation fold, BTTDA is fit to a prior determined maximum
    number of blocks with a given `theta` candidate. A line search over the
    number of blocks then determines the optimal `n_blocks` for this `theta` and
    the overal highest scoring pair `(theta, n_blocks)` can be picked using a grid search
    over `theta` candidates.

    In the end, this class acts as an applicable BTTDA instance with best
    hyperparameters determined during fitting

    Parameters
    ----------
    max_n_blocks : int
        The maximum amount of blocks to which to extend the line search for
        `n_blocks`.

    thetas : None or iterable of float, default=None
        `theta` candidates between 0 and 1 to try for the grid search.
        If None, use `[0.0, 0.1, 0.2, ... , 1.0]`.

    clf : sklearn.base.ClassifierMixin
        Classifier to determine the score of cross-validation folds. Must be
        specified.

    scorer : str, callable, list, tuple or dict, default=None
        Scoring strategy to evaluate cross-validation folds. See
        `sklearn.model_selection.GridSearchCV` for further details.
        For binary classification problems, `scorer` is set by default to
        'roc_auc' and for multiclass problems to 'accuracy'.

    cv : cross-validation generator, default=None
        Determines the cross-validation splitting strategy. If `cv`is None,
        `StratifiedKFold()` is used.

    n_jobs : int, default=1,
        Number of jobs to run in parallel. -1 means using all processors.
        None means using 1 processors unless in a joblib backend.

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

    fixed_n_blocks : bool, default=False
        If True, don't perform a line search over the number of blocks but only
        evaluate one candidate where `n_blocks` is `max_n_blocs`.
    """

    def __init__(
        self,
        max_n_blocks=16,
        thetas=None,
        clf=None,
        scorer=None,
        cv=None,
        n_jobs=1,
        hoda_params=None,
        extra_train_info=False,
        verbose=False,
        forward=True,
        fixed_n_blocks=False,
    ):
        self.max_n_blocks = max_n_blocks
        self.thetas = thetas
        self.clf = clf
        self.scorer = scorer
        self.cv = cv
        self.n_jobs = n_jobs
        self.hoda_params = hoda_params
        self.verbose = verbose
        self.extra_train_info = extra_train_info
        self.forward = forward
        self.fixed_n_blocks = fixed_n_blocks

    def fit(self, X, y=None, groups=None):
        """Fit the estimator to the data.

        Parameters
        ----------
        X : tensorly.tensor of shape (n_samples, dim_1, dim_2, ..., dim_K)
            Training data.

        y : array-like of shape (n_samples), default=None
            Class labels.

        groups : array-like of shape (n_samples) of type obj, default=None
            Optional group labels for grouped cv splitters.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Set thetas
        thetas = self.thetas
        if thetas is None:
            thetas = np.arange(0, 1 + 0.1, 0.1)
        # Set classifier
        clf = self.clf
        if clf is None:
            raise ValueError("Must specify a decision classifier")

        # Set scoring
        scorer = self.scorer
        n_classes = len(np.unique(y))
        if scorer is None:
            if n_classes > 2:
                scorer = get_scorer("accuracy")
            else:
                scorer = get_scorer("roc_auc")
        cv = self.cv
        if cv is None:
            cv = StratifiedKFold()

        self.results_ = self._gridsearch(X, y, thetas, clf, cv, scorer, groups)
        opt_theta, opt_n_blocks = (
            self.results_.groupby(["theta", "n_blocks"])
            .test_score.aggregate("mean")
            .idxmax()
        )

        if self.fixed_n_blocks:
            opt_n_blocks = self.max_n_blocks

        self.ranks = [None] * opt_n_blocks
        if self.hoda_params is None:
            self.hoda_params = dict()
        self.hoda_params["theta"] = opt_theta
        if self.verbose:
            print(f"Fitting BTTDA with theta={opt_theta}, n_blocks={opt_n_blocks}")
        return super().fit(X, y)

    def _make_bttda(self, theta):
        hoda_params = self.hoda_params
        if hoda_params is None:
            hoda_params = dict()
        hoda_params = hoda_params.copy()
        hoda_params["theta"] = theta
        bttda = BTTDA(
            ranks=[None] * self.max_n_blocks,
            hoda_params=hoda_params,
            verbose=self.verbose,
            extra_train_info=False,
            forward=False,
        )
        return bttda

    def _gridsearch(self, X, y, thetas, clf, cv, scorer, groups):
        args_list = []
        for fold, (train_idc, test_idc) in enumerate(cv.split(X, y, groups=groups)):
            for theta in thetas:
                args_list.append((X, y, fold, train_idc, test_idc, theta, clf, scorer))
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._eval_bttdacv_search_fold)(*args) for args in args_list
        )
        return pd.concat(results, ignore_index=True)

    def _eval_bttdacv_search_fold(
        self, X, y, fold, train_idc, test_idc, theta, clf, scorer
    ):
        if self.verbose:
            print(f"fold={fold}, theta={theta}")
        clf = clone(clf)
        bttda = self._make_bttda(theta)
        bttda.fit(X[train_idc], y[train_idc])
        result = []
        if self.fixed_n_blocks:
            n_blocks_range = [self.max_n_blocks]
        else:
            n_blocks_range = range(1, self.max_n_blocks + 1)
        for n_blocks in n_blocks_range:
            if n_blocks > bttda.n_blocks_:
                break
            if self.verbose:
                print(f"fold={fold}, theta={theta}, n_blocks={n_blocks}")
            Xt = bttda.transform(X, n_blocks=n_blocks)
            try:
                clf.fit(Xt[train_idc], y[train_idc])
                test_score = scorer(clf, Xt[test_idc], y[test_idc])
            except ValueError as e:
                warnings.warn(str(e))
                break

            result.append(
                dict(fold=fold, theta=theta, n_blocks=n_blocks, test_score=test_score)
            )
        result = pd.DataFrame(result)
        return result
