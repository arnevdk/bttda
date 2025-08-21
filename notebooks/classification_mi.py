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
from meeglet import define_frequencies, define_wavelets, plot_wavelet_family
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
            max_iter=256,
            toeplitz=(2,),
            taper=False,
            verbose=False,
            refit_shrinkage=True,
    )

def get_bttda_params():
    return dict(
        hoda_params=get_hoda_params(),
        verbose=False,
        cv=cv,
        n_jobs=-1,
        clf=make_clf()
    )

def stf_transform(X, sfreq=250, target_sfreq=32, f_min=8, f_max=32, n_freqs=16):
    """
    # define frequencies according to MEEGLET
    freqs, sigma_time, sigma_freq, bw_oct, qt = define_frequencies(
        foi_start=f_min, foi_end=f_max, bw_oct=0.5, delta_oct=1/8
    )
    n_cycles = freqs/2


    
    # perform time-frequency transform
    X_tfr = tfr_array_morlet(X, sfreq, freqs, n_cycles=n_cycles, zero_mean=True, output='complex', n_jobs=1)
    X_tfr = np.abs(X_tfr)
    
    X_tfr_base=  np.log(X_tfr)


    # Anti-alias and downsample
    for fi in range(X_tfr.shape[2]):
        X_tfr_base[:,:,fi,:] = filter_data(X_tfr_base[:,:,fi,:], sfreq, l_freq=None, h_freq=target_sfreq/2, verbose=False, n_jobs=1)
    decim = int(np.round(sfreq/target_sfreq))
    X_tfr_base = X_tfr_base[:,:,:,::decim]
    X_tfr_base = X_tfr_base[:,:,:,1:-1]

    return X_tfr_base
    """
    freqs = np.arange(f_min, f_max+1)
    wavelet = 'cmor6-1'
    center_freq = pywt.central_frequency(wavelet)
    scales = center_freq * sfreq / freqs
    coeffs, freqs_out = pywt.cwt(X, scales, wavelet, sampling_period=1/sfreq)
    X_tfr = np.abs(coeffs)**2
    X_tfr = np.moveaxis(X_tfr, 0,2)
    downsample_factor = 20
    n_bins = int(X_tfr.shape[-1]//20)
    X_tfr_sub = scipy.signal.resample(X_tfr, n_bins, axis=-1)
    return X_tfr_sub

    
def get_pipelines_mi():
    pipelines=dict()

    
    pipelines['HODA'] = Pipeline([
        ('stf', FunctionTransformer(stf_transform)),
        ('tensorly', FunctionTransformer(tl.tensor)),
        ('zscore1', ZScore()),
        ('bttda',BTTDACV(
            max_n_blocks=1,
            #thetas=[0.0 ,0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 1.0],
            thetas=[0.0 ,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            **get_bttda_params()
        )),
        ('clf', make_clf())
    ])
    


    """
    pipelines['PARAFACDA'] = Pipeline([
        ('stf', FunctionTransformer(stf_transform)),
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
        ('stf', FunctionTransformer(stf_transform)),
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
    """
    
    return pipelines