#!/usr/bin/env python3

def profile_hoda()
    import cupyx
    import tensorly as tl
    from hoda.hoda import HODA
    from mne.decoding import Scaler
    from moabb.datasets import BNCI2014_008
    from moabb.paradigms import P300
    
    tl.set_backend("cupy")
    
    tmin = 0
    tmax = 0.8
    fmin = 0.5
    fmax = 16
    sfreq = 32
    
    paradigm = P300(resample=sfreq, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
    dataset = BNCI2014_008()
    epochs, labels, meta = paradigm.get_data(
        dataset=dataset, subjects=[1], return_epochs=True
    )
    
    session = meta["session"][0]
    idc = meta["session"] == session
    epochs = epochs[idc]
    labels = labels[idc]
    meta = meta[idc]
    
    X = epochs.get_data()
    y = labels
    
    
    hoda = HODA(
        rank=3,
        max_iter=32,
        tol=1e-20,
        init="svd",
        shrinkage="lw",
        toeplitz=None,
        obj="rt",
        solver="lobpcg",
        taper=False,
        extra_train_info=False,
        verbose=False,
    )
    
    with cupyx.profiler.profile():
        hoda.fit(X, y)

if __name__ == "__main__":
    profile_hoda()
