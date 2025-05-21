import math
import pandas as pd
import pdb

import numpy as np
import tensorly as tl
from kneed import KneeLocator
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
import joblib
from hoda.tensorize import vec
from hoda.hoda import BTTDA, f_oneway
from sklearn.metrics import get_scorer
from sklearn.base import clone
from sklearn.feature_selection import SelectFdr


try:
    from toeplitzlda.classification import ToeplitzLDA
except ImportError:
    pass


class ZScore(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_ = tl.mean(X, axis=(0,-1))
        self.std_ = np.std(X, axis=(0,-1))
        return self

    def transform(self, X, y=None):
        shape = (1, *(X.shape[1:-1]), 1)
        mean = self.mean_.reshape(shape)
        std = self.std_.reshape(shape)
        return (X - mean) / std


class SelectFdrMin1(SelectFdr):

    def get_support(self, indices=False):
        support = super().get_support(indices=False)
        if not np.count_nonzero(support):
            support = np.zeros_like(support)
            support[np.argmax(self.scores_)] = True
        if indices:
            support = np.where(support)[0]
        return support


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
    ):
        self.max_n_blocks=max_n_blocks
        self.thetas=thetas
        self.clf=clf
        self.scorer=scorer
        self.cv = cv
        self.n_jobs=n_jobs
        self.hoda_params = hoda_params
        self.verbose = verbose
        self.extra_train_info = extra_train_info
        self.forward = forward

    def fit(self, X, y=None):
        # Set thetas
        thetas = self.thetas
        if thetas is None:
            thetas = np.arange(0,1+0.1, 0.1)
        # Set classifier
        clf = self.clf
        if clf is None:
            clf = self._make_default_clf()
        # Set scoring
        scorer = self.scorer
        if scorer is None:
            scorer = get_scorer('roc_auc')
        cv = self.cv
        if cv is None:
            cv = StratifiedKFold()

        self.results_ = self._gridsearch(X,y,thetas, clf, cv, scorer)
        opt_theta, opt_n_blocks = self.results_.groupby(['theta', 'n_blocks']).test_score.aggregate('mean').idxmax()
        
        self.ranks = [None]*opt_n_blocks
        if self.hoda_params is None:
            self.hoda_params = dict()
        self.hoda_params['theta'] = opt_theta
        if self.verbose:
            print(f'Fitting BTTDA with theta={opt_theta}, n_blocks={opt_n_blocks}')
        return super().fit(X,y)

    def _make_bttda(self, theta):
        hoda_params = self.hoda_params
        if hoda_params is None:
            hoda_params = dict()
        hoda_params = hoda_params.copy()
        hoda_params['theta'] = theta
        bttda = BTTDA(
                ranks=[None]*self.max_n_blocks,
                hoda_params=hoda_params,
                verbose=self.verbose,
                extra_train_info=False,
                forward=False
            )
        return bttda
 

    def _gridsearch(self, X,y, thetas, clf, cv, scorer):
        args_list = []
        for fold, (train_idc, test_idc) in enumerate(cv.split(X,y)):
            for theta in thetas:
                args_list.append((X,y,fold,train_idc, test_idc, theta, clf, scorer))
        #with joblib.parallel_backend('loky'):
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._eval_bttdacv_search_fold)(*args) for args in args_list)
        return pd.concat(results, ignore_index=True)

    def _eval_bttdacv_search_fold(self, X,y, fold,  train_idc, test_idc, theta, clf, scorer):
        if self.verbose:
            print(f'fold={fold}, theta={theta}')
        clf = clone(clf)
        bttda = self._make_bttda(theta)
        bttda.fit(X[train_idc], y[train_idc])
        result = []
        for n_blocks in range(1, self.max_n_blocks+1):
            if n_blocks > bttda.n_blocks_:
                break
            if self.verbose:
                print(f'fold={fold}, theta={theta}, n_blocks={n_blocks}')
            Xt = bttda.transform(X, n_blocks=n_blocks)
            try:
                clf.fit(Xt[train_idc], y[train_idc])
            except ValueError as e:
                break
            result.append(dict(
                fold=fold,
                theta=theta,
                n_blocks=n_blocks,
                test_score = scorer(clf, Xt[test_idc], y[test_idc])
            ))
        result = pd.DataFrame(result)
        return result
 


    def _make_default_clf(self):
        return make_pipeline(
            FunctionTransformer(tl.to_numpy),
            StandardScaler(),
            LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
        )

