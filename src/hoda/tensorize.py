import numpy as np
import scipy.linalg
import tensorly as tl
from mne.baseline import rescale
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
    y=None,
    morlet_params=None,
    #bin_freq=25,
    bin_freq=16,
    n_freqs = 16,
    zscore=True,
):
    # TFR
    if morlet_params is None:
        morlet_params = dict()
    morlet_params["output"] = "power"
    morlet_params.setdefault("freqs", np.geomspace(8, 32, n_freqs))
    morlet_params.setdefault("n_cycles", morlet_params["freqs"] * 0.7)
    Xt = np.abs(tfr_array_morlet(X, sfreq, **morlet_params))

    ## Baseline
    # if baseline_params is not None:
    #    baseline_params.setdefault("baseline", (-0.15, -0.05))
    #    baseline_params.setdefault("mode", "logratio")
    #    n_times = X.shape[-1]
    #    tmax = n_times / sfreq + tmin
    #    times = np.linspace(tmin, tmax, n_times)
    #    Xt = rescale(Xt, times, **baseline_params)

    # Time bins
    n_samples = X.shape[-1]
    epoch_len = n_samples / sfreq
    n_bins = int(bin_freq * epoch_len)
    Xt = np.apply_along_axis(
        lambda x: scipy.stats.binned_statistic(np.arange(len(x)), x, bins=n_bins)[0],
        -1,
        Xt,
    )

    if zscore:
        mean = Xt.mean(axis=0)
        std = Xt.std(axis=0)
        Xt = (Xt - mean) / std

    return Xt


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
