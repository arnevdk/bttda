try:
    import cupy.linalg
    import cupyx.scipy.linalg
    import cupyx.scipy.sparse.linalg
except ImportError:
    pass

import numpy as np
import scipy.linalg
import tensorly as tl


def pinvh(A):
    if tl.get_backend() == "cupy":
        return cupy.linalg.pinv(A)
    elif tl.get_backend() == "numpy":
        return scipy.linalg.pinvh(A)
    else:
        raise NotImplementedError


def trunc_svd(A, r=None):
    n = A.shape[0]
    if r is None:
        r = n
    if tl.get_backend() == "cupy":
        vl, w, vr = cupy.linalg.svd(A, full_matrices=False)
        vl = vl[:, :r]
        w = w[:r]
        vr = vr[:, :r]
    elif tl.get_backend() == "numpy":
        vl, w, vr = tl.partial_svd(A, n_eigenvecs=r)
    else:
        raise NotImplementedError
    return vl, w, vr


def trunc_gevd(A, B=None, r=None):
    n = A.shape[0]
    if r is None:
        r = n
    if tl.get_backend() == "numpy":
        subset = [n - r, n - 1]
        w, v = scipy.linalg.eigh(A, B, subset_by_index=subset)
        w = w[::-1]
        v = v[:, ::-1]
    else:
        raise NotImplementedError
    return v, w


def trunc_eigh(A, r=None):
    if tl.get_backend() == "cupy":
        w, v = cupy.linalg.eigh(A)
        idc = np.argsort(-tl.abs(w))[:r]
        w = w[idc]
        v = v[:, idc]
    else:
        # TODO
        raise NotImplementedError
    return v, w


def lobpcg(*args, **kwargs):
    if tl.get_backend() == "cupy":
        return cupyx.scipy.sparse.linalg.lobpcg(*args, **kwargs)
    elif tl.get_backend() == "numpy":
        return scipy.sparse.linalg.lobpcg(*args, **kwargs)
    else:
        raise NotImplementedError


def toeplitz(a):
    if tl.get_backend() == "numpy":
        A = scipy.linalg.toeplitz(a)
    elif tl.get_backend() == "cupy":
        A = cupyx.scipy.linalg.toeplitz(a)
    else:
        raise NotImplementedError
    return A


def force_toeplitz(A, taper=False):
    n, _ = A.shape
    toep = tl.zeros(n)
    for i in range(n):
        diag_i = tl.diag(A, k=i)
        mean_diag_i = tl.mean(diag_i)
        toep[i] = mean_diag_i
    if taper:
        taper = tl.arange(len(toep), 0, -1) - 1
        toep = toep * taper
    cov_toep = toeplitz(toep)
    return cov_toep


def det(A):
    if tl.get_backend() == "numpy":
        return scipy.linalg.det(A)
    elif tl.get_backend() == "cupy":
        return cupy.linalg.det(A)
    else:
        raise NotImplementedError
