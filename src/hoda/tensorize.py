import numpy as np
import scipy.linalg
from mne.baseline import rescale
from mne.time_frequency import tfr_array_morlet
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer


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
    tmin=0,
    y=None,
    morlet_params=None,
    baseline_params=None,
    decim=1,
):
    # TFR
    if morlet_params is None:
        morlet_params = dict()
    morlet_params["output"] = "power"
    morlet_params.setdefault("freqs", np.linspace(0.5, 16, 16))
    morlet_params.setdefault("n_cycles", morlet_params["freqs"] * 0.7)
    Xt = tfr_array_morlet(X, sfreq, **morlet_params)

    # Baseline
    if baseline_params is not None:
        baseline_params.setdefault("baseline", (-0.15, -0.05))
        baseline_params.setdefault("mode", "logratio")
        n_times = X.shape[-1]
        tmax = n_times / sfreq + tmin
        times = np.linspace(tmin, tmax, n_times)
        Xt = rescale(Xt, times, **baseline_params)
    # Decimate
    Xt = Xt[:, :, :, ::decim]
    return Xt


def crop(X, begin=0, end=None, sfreq=None, tmin=None):
    begin = int((begin - tmin) * sfreq)
    if end is None:
        return X[..., begin:]
    else:
        end = int((end - tmin) * sfreq)
        return X[..., begin:end]


class HankelTensor(FunctionTransformer):
    def __init__(**params):
        params["func"] = hankel_tensor
        super().__init__(**params)


def STFTensor(**params):
    return FunctionTransformer(stf_tensor, kw_args=params)


def Crop(**params):
    return FunctionTransformer(crop, kw_args=params)
