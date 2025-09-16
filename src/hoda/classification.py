import math
import pdb
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorly as tl
from joblib import Parallel, delayed
from kneed import KneeLocator
from sklearn.base import (BaseEstimator, ClassifierMixin, TransformerMixin,
                          clone)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFdr, f_classif
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from hoda.hoda import BTTDA, f_oneway
from hoda.tensorize import vec

try:
    from toeplitzlda.classification import ToeplitzLDA
except ImportError:
    pass


class ZScore(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_ = tl.mean(X, axis=(0, -1))
        self.std_ = np.std(X, axis=(0, -1))
        return self

    def transform(self, X, y=None):
        shape = (1, *(X.shape[1:-1]), 1)
        mean = self.mean_.reshape(shape)
        std = self.std_.reshape(shape)
        return (X - mean) / std


class ZLogRatio(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=(0, -1))
        self.std_log_ = np.std(np.log10(X), axis=(0, -1))
        return self

    def transform(self, X, y=None):
        shape = (1, *(X.shape[1:-1]), 1)
        mean = self.mean_.reshape(shape)
        std_log = self.std_log_.reshape(shape)
        return np.log10(X / mean) / std_log


class SelectFCutoff(BaseEstimator, TransformerMixin):

    def __init__(self, cutoff=1):
        self.cutoff = cutoff

    def fit(self, X, y=None):
        self.scores_, self.p_values_ = f_classif(X, y)
        self.scores_[np.isinf(self.scores_)] = 0
        self.scores_[np.isnan(self.scores_)] = 0
        self.support_ = self.scores_ > self.cutoff
        if not np.any(self.support_):
            self.support_ = np.zeros(len(self.support_), dtype=bool)
            self.support_[np.argmax(self.scores_)] = True
        return self

    def transform(self, X, y=None):
        return X[:, self.support_]


class SelectFdrMin1(SelectFdr):

    def get_support(self, indices=False):
        support = super().get_support(indices=False)
        if not np.count_nonzero(support):
            support = np.zeros_like(support)
            support[np.argmax(self.scores_)] = True
        if indices:
            support = np.where(support)[0]
        return support


class SelectFKneepoint(TransformerMixin):

    def fit(self, X, y):
        f, p = f_classif(X, y)

        return self


class ToeplitzLDAWrapper(BaseEstimator, ClassifierMixin):
    def fit(self, X, y=None):
        self.classes_ = np.unique(y)
        n_epochs, n_channels, n_samples = X.shape
        self.tlda_ = ToeplitzLDA(n_channels=n_channels, data_is_channel_prime=False)
        return self.tlda_.fit(vec(X), y)

    def decision_function(self, X):
        n_epochs, n_channels, n_samples = X.shape
        return self.tlda_.decision_function(vec(X))

    def predict(self, X):
        return self.tlda_.predict(vec(X))

    def predict_proba(self, X):
        return self.tlda_.predict_proba(vec(X))


class BTTDACV(BTTDA):
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

    def fit(self, X, y=None):
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

        self.results_ = self._gridsearch(X, y, thetas, clf, cv, scorer)
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

    def _gridsearch(self, X, y, thetas, clf, cv, scorer):
        args_list = []
        for fold, (train_idc, test_idc) in enumerate(cv.split(X, y)):
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
