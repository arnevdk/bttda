from sklearn.utils import safe_mask, safe_sqr

try:
    import cupy.linalg
    import cupyx.scipy.linalg
    import cupyx.scipy.sparse.linalg
    import cupyx.scipy.special
except ImportError:
    pass

import numpy as np
import scipy.linalg
import scipy.special
import tensorly as tl


def pinvh(A):
    if tl.get_backend() == "cupy":
        return cupy.linalg.pinv(A)
    elif tl.get_backend() == "numpy":
        return scipy.linalg.pinvh(A)
    else:
        raise NotImplementedError


def solve(A, B):
    if tl.get_backend() == "cupy":
        return cupy.linalg.solve(A, B)
    elif tl.get_backend() == "numpy":
        return scipy.linalg.solve(A, B)
    else:
        raise NotImplementedError


def trunc_eigh(
    A, B=None, rank=None, largest=True, init=None, method="lanczos", solver_params=None
):
    """

    SVD eigensolver can only be used if  B^-1@A is semi-positive definite
    """
    if init is None:
        init = tl.eye(A.shape, dtype=A.dtype)
    init = init[:, :rank]
    solver_params = solver_params or dict()
    if method == "lanczos":
        v, w = lanczos(A, B=B, rank=rank, largest=largest, **solver_params)
    elif method == "svd":
        solver_params["flip_sign"] = True
        solver_params.setdefault("method", "truncated_svd")
        if largest:
            solver_params["n_eigenvecs"] = rank
        else:
            solver_params["n_eigenvecs"] = None
        if B is None:
            v, w, _ = tl.tenalg.svd_interface(A, **solver_params)
        else:
            v, w, _ = tl.tenalg.svd_interface(solve(B @ A), **solver_params)
        if not largest:
            w = w[-rank:]
            v = v[:, -rank:]

    elif method == "lobpcg":
        v, w = lobpcg(A, init, rank=rank, largest=largest, B=B)
    return v, w


def lanczos(A, B=None, rank=None, largest=True, **kwargs):
    if rank is None:
        rank = A.shape[0]
    if tl.get_backend() == "cupy":
        if B is not None:
            M = solve(B, A)
        else:
            M = A
        # if rank is None or rank == M.shape[0]:
        #    w, v = cupy.linalg.eigh(M)
        # else:
        #    which = "LA" if largest else "SA"
        #    w, v = cupyx.scipy.sparse.linalg.eigsh(
        #        M, k=rank, return_eigenvectors=True, which=which, **kwargs
        #    )
        w, v = cupy.linalg.eigh(M)
        if largest:
            w = w[-rank:]
            v = v[:, -rank:]
        else:
            w = w[:rank]
            v = v[:, :rank]
    elif tl.get_backend() == "numpy":
        if largest:
            n = A.shape[0]
            subset = [n - rank, n - 1]
        else:
            subset = [0, rank - 1]
        w, v = scipy.linalg.eigh(A, b=B, subset_by_index=subset, **kwargs)
    else:
        raise NotImplementedError
    return v, w


def lobpcg(A, init, B=None, rank=None, **kwargs):
    if tl.get_backend() == "cupy":
        w, v = cupyx.scipy.sparse.linalg.lobpcg(A, init, B=B, **kwargs)
    elif tl.get_backend() == "numpy":
        w, v = scipy.sparse.linalg.lobpcg(A, init, B=B, **kwargs)
    else:
        raise NotImplementedError
    return v, w


def toeplitz(a):
    if tl.get_backend() == "numpy":
        A = scipy.linalg.toeplitz(a)
    elif tl.get_backend() == "cupy":
        A = cupyx.scipy.linalg.toeplitz(a)
    else:
        raise NotImplementedError
    return A


def det(A):
    if tl.get_backend() == "numpy":
        return scipy.linalg.det(A)
    elif tl.get_backend() == "cupy":
        return cupy.linalg.det(A)
    else:
        raise NotImplementedError


def maximum(A, B):
    if tl.get_backend() == "numpy":
        return np.maximum(A, B)
    elif tl.get_backend() == "cupy":
        return cupy.maximum(A, B)
    else:
        raise NotImplementedError


def chdtrc(v, x):
    if tl.get_backend() == "numpy":
        return scipy.special.chdtrc(v, x)
    elif tl.get_backend() == "cupy":
        return cupyx.scipy.special.chdtrc(v, x)
    else:
        raise NotImplementedError


def fdtrc(dfn, dfd, x):
    if tl.get_backend() == "numpy":
        return scipy.special.fdtrc(dfn, dfd, x)
    elif tl.get_backend() == "cupy":
        return cupyx.scipy.special.fdtrc(dfn, dfd, x)
    else:
        raise NotImplementedError
