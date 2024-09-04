import numpy as np
import scipy.linalg
import scipy.stats
import tensorly as tl
from mne.baseline import rescale
from mne.filter import filter_data
from mne.time_frequency import tfr_array_morlet
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def vec(X, y=None, extra=None):
    x = X.reshape((X.shape[0], -1))
    return x


def hankel_tensor(X, y=None):
    n_times = X.shape[-1]
    half = n_times // 2 + 1

    def _hankel_nopad(x):
        return scipy.linalg.hankel(x[:half], x[half - 1 :])

    return np.apply_along_axis(_hankel_nopad, -1, X)


def hankel_tensor_inv(Xh, y=None):
    X1 = Xh[:, :, 0, :]
    X2 = Xh[:, :, -1, :]
    if Xh.shape[-1] == Xh.shape[-2]:
        X2 = X2[:, :, 1:]
    return np.concatenate((X1, X2), axis=-1)


def stf_tensor(
    X,
    sfreq=None,
    morlet_params=None,
    baseline_params=None,
    l_freq=8,
    h_freq=32,
    bin_freq=4,
    n_freqs=4,
    normalize=True,
    log=True,
):
    # TFR morlet
    # if morlet_params is None:
    #    morlet_params = dict()
    # morlet_params["output"] = "complex"
    # morlet_params.setdefault("freqs", np.geomspace(8, 32, n_freqs))
    # morlet_params.setdefault("n_cycles", morlet_params["freqs"] * 0.7)
    # X_tfr = np.abs(tfr_array_morlet(X, sfreq, **morlet_params))

    # TFR filter-hilbert
    n_samples, n_channels, n_times = X.shape
    X_tfr = np.zeros((n_samples, n_channels, n_freqs, n_times))
    freqs = np.geomspace(l_freq, h_freq, n_freqs + 1)
    for f in range(len(freqs) - 1):
        xf = filter_data(X, sfreq, freqs[f], freqs[f + 1])
        xf = scipy.signal.hilbert(xf)
        xf = np.abs(xf)
        X_tfr[:, :, f, :] = xf

    # Time bins
    n_samples = X.shape[-1]
    epoch_len = n_samples / sfreq
    n_bins = int(bin_freq * epoch_len)
    X_tfr = np.apply_along_axis(
        lambda x: scipy.stats.binned_statistic(np.arange(len(x)), x, bins=n_bins)[0],
        -1,
        X_tfr,
    )

    if log:
        X_tfr = np.log(X_tfr)

    # Zscore
    if normalize:
        # mu = Xt.mean(axis=(0, 1, 3))[np.newaxis, np.newaxis, :, np.newaxis]
        # sigma = Xt.std(axis=(0, 1, 3))[np.newaxis, np.newaxis, :, np.newaxis]
        mu = X_tfr.mean(axis=0)[np.newaxis]
        sigma = X_tfr.std(axis=0)[np.newaxis]
        X_tfr = (X_tfr - mu) / sigma
    return X_tfr


def crop(X, begin=0, end=None, sfreq=None, tmin=None):
    begin = int((begin - tmin) * sfreq)
    if end is None:
        return X[..., begin:]
    else:
        end = int((end - tmin) * sfreq)
        return X[..., begin:end]


class Vectorize(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return tl.to_numpy(vec(X))


class Tensorize(TransformerMixin):
    def __init__(self, method=None, params=None):
        self.method = method
        self.params = params

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        params = self.params
        if params is None:
            params = dict()
        if self.method == "stf":
            X = stf_tensor(X, **params)
        elif self.method == "hankel":
            X = hankel_tensor(X, **params)
        if not tl.is_tensor(X):
            X = tl.tensor(X)
        return X


def Crop(**params):
    return FunctionTransformer(crop, kw_args=params)
