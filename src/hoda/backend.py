from sklearn.utils import safe_mask, safe_sqr

try:
    import cupy
    import cupyx.scipy
    import cupyx.scipy.linalg
except ImportError:
    pass

import numpy as numpy_mod
import scipy as scipy_mod
import tensorly as tl


class Backend(object):
    def __init__(self, module):
        self._module = module

    def __getattr__(self, attr):
        """Only called if attribute does not exist"""
        backend_attr = getattr(self._module, attr, None)
        if backend_attr is None:
            raise NotImplementedError(f"{self._backend_module}.{attr}")
        return backend_attr


class NumpyBackend(Backend):
    def __init__(self):
        super().__init__(numpy_mod)


class CupyBackend(Backend):
    def __init__(self):
        super().__init__(cupy)


class ScipyBackend(Backend):
    def __init__(self):
        super().__init__(scipy_mod)

    @staticmethod
    def lanczos(A, B=None, rank=None, largest=True, **kwargs):
        if largest:
            n = A.shape[0]
            subset = [n - rank, n - 1]
        else:
            subset = [0, rank - 1]
        w, v = scipy_mod.linalg.eigh(A, b=B, subset_by_index=subset, **kwargs)
        return v, w


class CupyxScipyBackend(Backend):
    def __init__(self):
        super().__init__(cupyx.scipy)

    @staticmethod
    def lanczos(A, B=None, rank=None, largest=True, **kwargs):
        if rank is None:
            rank = A.shape[0]
        if B is not None:
            # https://discuss.tensorflow.org/t/compute-generalised-eigenvectors/12323
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


def _numpy_backend():
    backend = tl.get_backend()
    if backend == "cupy":
        return CupyBackend()
    elif backend == "numpy":
        return NumpyBackend()
    else:
        raise NotImplementedError


def _scipy_backend():
    backend = tl.get_backend()
    if backend == "numpy":
        return ScipyBackend()
    elif backend == "cupy":
        return CupyxScipyBackend()
    else:
        raise NotImplementedError


def __getattr__(name):
    if name == "np" or name == "numpy":
        return _numpy_backend()
    if name == "scipy":
        return _scipy_backend()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
