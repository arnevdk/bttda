from moabb.datasets import *
from moabb.paradigms import P300
from hoda import HODA
import numpy as np

from mne.time_frequency import tfr_morlet
from mne.time_frequency.tfr import morlet

from sklearn.utils.class_weight import compute_sample_weight
paradigm = P300(tmin=-3.2, tmax=4, resample=None, fmax=None, fmin=0.1)
dataset = BNCI2014008()
subject, session, run = 1, "session_0", "run_0"
dataset.download(subject_list=[subject])
data = dataset.get_data(subjects=[subject])
raw = data[subject][session][run]
epochs, labels, _ = paradigm.process_raw(raw, dataset, return_epochs=True)
n_freqs=8
sfreq=8
freqs= np.geomspace(1, 16, n_freqs)
n_cycles = np.linspace(2, 4, n_freqs)
wavelets = morlet(epochs.info['sfreq'],
                  freqs, n_cycles, zero_mean=True)
epochs = tfr_morlet(epochs, freqs, n_cycles, return_itc=False, average=False,
                    output='complex', verbose=True, n_jobs=12, decim=int(epochs.info['sfreq']//sfreq))
epochs.data -= np.average(epochs.data, weights=compute_sample_weight('balanced', labels), axis=0)
epochs.data /= np.std(epochs.data, axis=0)
epochs.crop(0, 1)
X = epochs.data
y = labels
hoda = HODA(max_iter=64, rank=(3,3,3), tol=0, initialize ='random', verbose=True,
            toeplitz=2, taper=None, shrinkage='oas', solver='eig')
hoda.fit(X,y)