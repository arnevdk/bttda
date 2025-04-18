from moabb.datasets import *
from moabb.paradigms import P300,LeftRightImagery
import tensorly as tl
import tensorly as tl
import numpy as np
import scipy.stats
from hoda.classification import ZScore
from mne.filter import resample
import scipy.stats.mstats

def load_moabb_p300(dataset, subjects=None, session=None, tmin=0):
    paradigm = P300(resample=48, tmin=tmin)
    dataset = dataset()
    epochs, labels, meta = paradigm.get_data(
        dataset=dataset, 
         subjects=subjects,
         return_epochs=True
    )
    if session is not None:
        session = meta['session'][session]
        idc = meta['session'] == session
        epochs = epochs[idc]
        labels = labels[idc]
        meta = meta[idc]
    return epochs, labels, meta

def fh_envelope(X, sfreq=250, target_sfreq=32):
    X = X.transpose((0,3,1,2))
    X = np.log(np.abs(scipy.signal.hilbert(X)))
    X = resample(X, down=sfreq/target_sfreq, n_jobs=-1, verbose=True)
    return X

def postprocess(X,y, standardize=True, winsorize=False):
    if winsorize:
        X = scipy.stats.mstats.winsorize(X)
    print(X.shape)
    X = tl.tensor(X)
    if standardize:
        X = ZScore().fit_transform(X,y)
    
    return X
