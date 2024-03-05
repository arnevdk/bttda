from sklearn.utils import safe_mask, safe_sqr

try:
    import cupy
    import cupyx.scipy
    import cupyx.scipy.linalg
    import cupyx.scipy.sparse.linalg
    import cupyx.scipy.special
except ImportError:
    pass

try:
    import torch
    import torch.linalg
except ImportError:
    pass

import scipy.linalg
import scipy.special
import tensorly as tl


def lanczos(*args, **kwargs):
    if tl.get_backend() == "numpy":
        return lanczos_numpy(*args, **kwargs)
    elif tl.get_backend() == "cupy":
        return lanczos_cupy(*args, **kwargs)
    elif tl.get_backend() == "pytorch":
        return lanczos_numpy(*args, **kwargs)
    else:
        raise NotImplementedError


def lanczos_numpy(
    A, B=None, rank=None, largest=True, init=None, force_spd=False, **kwargs
):
    if largest:
        n = A.shape[0]
        subset = [n - rank, n - 1]
    else:
        subset = [0, rank - 1]
    w, v = scipy.linalg.eigh(A, b=B, subset_by_index=subset, **kwargs)
    return v, w


def lanczos_cupy(A, B=None, rank=None, largest=True, force_spd=False, init=None):
    if rank is None:
        rank = A.shape[0]
    if B is not None:
        # https://discuss.tensorflow.org/t/compute-generalised-eigenvectors/12323

        if force_spd:
            # Ensure B is semi-positive definite
            eigvals, eigvecs = cupy.linalg.eigh(B)
            eigvals[eigvals < 0] = 0
            eigvecs, _ = tl.qr(eigvecs, mode="complete")
            _, L = cupy.linalg.qr(
                cupy.diag(cupy.sqrt(eigvals)) @ eigvecs.T, mode="complete"
            )
            L = L.T
        else:
            L = cupy.linalg.cholesky(B)

        Y = cupyx.scipy.linalg.solve_triangular(L, A.T, lower=True).T
        C = cupyx.scipy.linalg.solve_triangular(L, Y, lower=True)
    else:
        C = A
    w, v = cupy.linalg.eigh(C)
    if B is not None:
        v = cupyx.scipy.linalg.solve_triangular(L.T, v, lower=False)

    if largest:
        w = w[-rank:]
        v = v[:, -rank:]
    else:
        w = w[:rank]
        v = v[:, :rank]
    return v, w


def lanczos_pytorch(A, B=None, rank=None, largest=True, force_spd=False, init=None):
    if rank is None:
        rank = A.shape[0]
    if B is not None:
        # https://discuss.tensorflow.org/t/compute-generalised-eigenvectors/12323

        if force_spd:
            # Ensure B is semi-positive definite
            eigvals, eigvecs = torch.linalg.eigh(B)
            eigvals[eigvals < 0] = 0
            eigvecs, _ = torch.linalg.qr(eigvecs, mode="complete")
            _, L = torch.linalg.qr(
                tl.diag(tl.sqrt(eigvals)) @ eigvecs.T, mode="complete"
            )
            L = L.T
        else:
            L = torch.linalg.cholesky(B)

        Y = torch.linalg.solve_triangular(L, A.T, lower=True).T
        C = torch.linalg.solve_triangular(L, Y, lower=True)
    else:
        C = A
    w, v = torch.linalg.eigh(C)
    if B is not None:
        v = torch.linalg.solve_triangular(L.T, v, lower=False)

    if largest:
        w = w[-rank:]
        v = v[:, -rank:]
    else:
        w = w[:rank]
        v = v[:, :rank]
    return tl.tensor(v), tl.tensor(w)


def lobpcg(A, X, B=None, M=None, *args, **kwargs):
    if tl.get_backend() == "numpy":
        return scipy.sparse.linalg.lobpcg(A, X, *args, B=B, M=M, **kwargs)
    elif tl.get_backend() == "cupy":
        return cupyx.scipy.sparse.linalg.lobpcg(A, X, *args, B=B, M=M, **kwargs)
    elif tl.get_backend() == "pytorch":
        return torch.lobpcg(A, *args, B=B, X=X, iK=M, **kwargs)
    else:
        raise NotImplementedError


def toeplitz(*args, **kwargs):
    if tl.get_backend() == "numpy":
        return scipy.linalg.toeplitz(*args, **kwargs)
    elif tl.get_backend() == "cupy":
        return cupyx.scipy.linalg.toeplitz(*args, **kwargs)
    else:
        raise NotImplementedError


def fdtrc(*args):
    if tl.get_backend() == "numpy":
        return scipy.special.fdtrc(*args)
    elif tl.get_backend() == "cupy":
        return scipy.special.fdtrc(*args)
    else:
        raise NotImplementedError


def chdtrc(*args):
    if tl.get_backend() == "numpy":
        return scipy.special.chdtrc(*args)
    elif tl.get_backend() == "cupy":
        return scipy.special.chdtrc(*args)
    else:
        raise NotImplementedError


def copy(A):
    if tl.get_backend() == "pytorch":
        return A.detach().clone()
    else:
        return A.copy()
