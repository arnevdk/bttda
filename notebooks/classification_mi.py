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
from mne.time_frequency import tfr_array_morlet
import numpy as np
import pywt
import scipy.signal

from mne.filter import filter_data

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
            max_iter=512,
            toeplitz=None,
            taper=False,
            verbose=False,
            refit_shrinkage=True,
            tol=1e-4,
    )

def get_bttda_params():
    return dict(
        hoda_params=get_hoda_params(),
        verbose=False,
        cv=cv,
        n_jobs=1,
        clf=make_clf()
    )

def stf_transform(X, sfreq=250, downsample_factor=20, f_min=8, f_max=32, n_freqs=16):
 
    freqs = np.geomspace(f_min, f_max, n_freqs)
    wavelet = 'cmor6-1'
    center_freq = pywt.central_frequency(wavelet)
    scales = center_freq * sfreq / freqs
    coeffs, freqs_out = pywt.cwt(X, scales, wavelet, sampling_period=1/sfreq)
    coeffs = np.moveaxis(coeffs, 0,2)
    X_tfr = np.log(np.abs(coeffs))
    n_bins = int(X_tfr.shape[-1]//downsample_factor)
    X_tfr_sub = scipy.signal.resample(X_tfr, n_bins, axis=-1)
    X_tfr_sub = X_tfr_sub[:,:,:,1:-1]
    return X_tfr_sub

    
def get_pipelines_mi():
    pipelines=dict()

    
    pipelines['HODA'] = Pipeline([
        #('tensorly', FunctionTransformer(tl.tensor)),
        ('zscore1', ZScore()),
        ('bttda',BTTDACV(
            max_n_blocks=1,
            #thetas=[0.0 ,0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 1.0],
            thetas=[0.0 ,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            **get_bttda_params()
        )),
        ('clf', make_clf())
    ])
    



    pipelines['PARAFACDA'] = Pipeline([
        ('tensorly', FunctionTransformer(tl.tensor)),
        ('zscore1', ZScore()),
        ('bttda',BTTDACV(
            max_n_blocks=16,
            thetas=[0],
            **get_bttda_params()
        )),
        ('clf', make_clf())
    ])
    
    pipelines['BTTDA'] = Pipeline([
        ('tensorly', FunctionTransformer(tl.tensor)),
        ('zscore1', ZScore()),
        ('bttda',BTTDACV(
            max_n_blocks=16,
            #thetas=[0.0 ,0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 1.0],
            thetas=[0.0 ,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            **get_bttda_params()
        )),
        ('clf', make_clf())
    ])

    
    return pipelines