import unittest

import numpy as np
import tensorly as tl
from hoda.hoda import HODA, trunc_eigh


class TestAlgorithmIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)

    def test_hoda(self):
        N = 1000
        X = np.random.rand(N, 32, 32)
        y = np.zeros(N)
        y[500:] = 1
        hoda = HODA()
        tl.set_backend("numpy", local_threadsafe=True)
        Xt_numpy = hoda.fit_transform(X, y)
        tl.set_backend("cupy", local_threadsafe=True)
        Xt_cupy = hoda.fit_transform(X, y)
        Xt_cupy = tl.to_numpy(Xt_cupy)
        np.testing.assert_allclose(Xt_numpy, Xt_cupy)


class TestTenalg(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        # SPD matrices, does not work for generic symetric matrices
        A = np.random.rand(32, 32)
        cls.A = A @ A.T
        B = np.random.rand(32, 32)
        cls.B = B @ B.T

    def _backend_trunc_eigh(self, backend, A, B=None, **kwargs):
        tl.set_backend(backend, local_threadsafe=True)
        A = tl.tensor(A)
        if B is not None:
            B = tl.tensor(B)
        v, w = trunc_eigh(A, B=B, **kwargs)
        v = tl.to_numpy(v)
        w = tl.to_numpy(w)
        return v, w

    def _test_trunc_eigh(self, *args, **kwargs):
        v_numpy, w_numpy = self._backend_trunc_eigh("numpy", *args, **kwargs)
        v_cupy, w_cupy = self._backend_trunc_eigh("cupy", *args, **kwargs)
        np.testing.assert_allclose(w_numpy, w_cupy, rtol=0, atol=1e-4)
        np.testing.assert_allclose(v_numpy, v_cupy, rtol=0, atol=1e-4)

    def test_lanczos_evd_full(self):
        self._test_trunc_eigh(self.A, method="lanczos")

    def test_lanczos_evd_trunc_largest(self):
        self._test_trunc_eigh(self.A, method="lanczos", rank=16, largest=True)

    def test_lanczos_evd_trunc_smallest(self):
        self._test_trunc_eigh(self.A, method="lanczos", rank=16, largest=False)

    def test_lanczos_gevd_full(self):
        self._test_trunc_eigh(self.A, B=self.B, method="lanczos")

    def test_lanczos_gevd_trunc_largest(self):
        self._test_trunc_eigh(self.A, B=self.B, method="lanczos", rank=16, largest=True)

    def test_lanczos_gevd_trunc_smallest(self):
        self._test_trunc_eigh(
            self.A, B=self.B, method="lanczos", rank=16, largest=False
        )

    # TODO test non SPD


if __name__ == "__main__":
    unittest.main()
