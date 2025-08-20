from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from hoda.classification import ZScore, BTTDACV, SelectFCutoff
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import tensorly as tl
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from meeglet import define_frequencies, define_wavelets, plot_wavelet_family
import matplotlib
import numpy as np

cv=StratifiedKFold(random_state=42, shuffle=True)



def make_clf():
    return  make_pipeline(
        FunctionTransformer(tl.to_numpy),
        PCA(n_components=None, whiten=True),
        SelectFCutoff(cutoff=1),
        LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
    )

def get_hoda_params():
    return dict(
            max_iter=128,
            toeplitz=(1,),
            taper=False,
            verbose=False,
            refit_shrinkage=True,
    )

def get_bttda_params():
    return dict(
        hoda_params=get_hoda_params(),
        verbose=False,
        cv=cv,
        n_jobs=5*11,
        clf = make_clf()

    )



def get_pipelines():

    pipelines=dict()

    pipelines['HODA'] = Pipeline([
        ('tensorly', FunctionTransformer(tl.tensor)),
        ('zscore1', ZScore()),
        ('bttda',BTTDACV(
            max_n_blocks=1,
            thetas=[0,0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8, 0.9,1],
            **get_bttda_params())
        ),
        ('clf', make_clf())
    ])

    pipelines['PARAFACDA'] = Pipeline([
        ('tensorly', FunctionTransformer(tl.tensor)),
        ('zscore1', ZScore()),
        ('bttda',BTTDACV(
            max_n_blocks=16,
            thetas=[0],
            **get_bttda_params())
        ),
        ('clf', make_clf())
    ])

    pipelines['BTTDA'] = Pipeline([
        ('tensorly', FunctionTransformer(tl.tensor)),
        ('zscore1', ZScore()),
        ('bttda',BTTDACV(
            max_n_blocks=16,
            thetas=[0,0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8, 0.9,1],
            **get_bttda_params())
        ),
        ('clf', make_clf())
    ])
    return pipelines