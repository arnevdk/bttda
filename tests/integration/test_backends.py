import unittest

import numpy as np
import tensorly as tl
from bttda.hoda import HODA
from bttda.util import center, solve_gevdh, toeplitz
from sklearn.datasets import make_spd_matrix


class TestAlgorithmIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)

    def test_hoda(self):
        N = 1000
        X = np.random.rand(N, 32, 32)
        y = np.zeros(N)
        y[500:] = 1
        hoda = HODA(solver="lobpcg", theta=0.5)
        tl.set_backend("numpy", local_threadsafe=True)
        Xt_numpy = hoda.fit_transform(tl.tensor(X), y)
        tl.set_backend("cupy", local_threadsafe=True)
        Xt_cupy = hoda.fit_transform(tl.tensor(X), y)
        Xt_cupy = tl.to_numpy(Xt_cupy)
        np.testing.assert_allclose(Xt_numpy, Xt_cupy, atol=1e-5, rtol=0)


class TestGEVD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A = make_spd_matrix(32, random_state=1)
        cls.B = make_spd_matrix(32, random_state=2)

    def _backend_solve_gevdh(self, backend, A, B=None, **kwargs):
        old_backend = tl.get_backend()
        tl.set_backend(backend, local_threadsafe=True)
        A = tl.tensor(A)
        if B is not None:
            B = tl.tensor(B)
        v, w = solve_gevdh(A, B=B, **kwargs)
        v = tl.to_numpy(v)
        w = tl.to_numpy(w)
        tl.set_backend(old_backend, local_threadsafe=True)
        return v, w

    def _test_solve_gevdh(self, *args, **kwargs):
        v_numpy, w_numpy = self._backend_solve_gevdh("numpy", *args, **kwargs)
        v_cupy, w_cupy = self._backend_solve_gevdh("cupy", *args, **kwargs)
        np.testing.assert_allclose(w_numpy, w_cupy, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(v_numpy, v_cupy, rtol=1e-4, atol=1e-4)

    def test_lanczos_evd_full(self):
        self._test_solve_gevdh(self.A, solver="lanczos")

    def test_lanczos_evd_SM(self):
        self._test_solve_gevdh(self.A, solver="lanczos", rank=16, which="SM")

    def test_lanczos_evd_LM(self):
        self._test_solve_gevdh(self.A, solver="lanczos", rank=16, which="LM")

    def test_lanczos_evd_SA(self):
        self._test_solve_gevdh(self.A, solver="lanczos", rank=16, which="SA")

    def test_lanczos_evd_LA(self):
        self._test_solve_gevdh(self.A, solver="lanczos", rank=16, which="LA")

    def test_lobpcg_evd_full(self):
        self._test_solve_gevdh(self.A, solver="lobpcg")

    def test_lobpcg_evd_SM(self):
        self._test_solve_gevdh(self.A, solver="lobpcg", rank=16, which="SM")

    def test_lobpcg_evd_LM(self):
        self._test_solve_gevdh(self.A, solver="lobpcg", rank=16, which="LM")

    def test_lobpcg_evd_SA(self):
        self._test_solve_gevdh(self.A, solver="lobpcg", rank=16, which="SA")

    def test_lobpcg_evd_LA(self):
        self._test_solve_gevdh(self.A, solver="lobpcg", rank=16, which="LA")

    def test_lobpcg_gevdh_full(self):
        self._test_solve_gevdh(self.A, B=self.B, solver="lobpcg")

    def test_lobpcg_gevdh_SM(self):
        self._test_solve_gevdh(self.A, B=self.B, solver="lobpcg", rank=16, which="SM")

    def test_lobpcg_gevdh_LM(self):
        self._test_solve_gevdh(self.A, B=self.B, solver="lobpcg", rank=16, which="LM")

    def test_lobpcg_gevdh_SA(self):
        self._test_solve_gevdh(self.A, B=self.B, solver="lobpcg", rank=16, which="SA")

    def test_lobpcg_gevdh_LA(self):
        self._test_solve_gevdh(self.A, B=self.B, solver="lobpcg", rank=16, which="LA")


class TestUtil(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.A = make_spd_matrix(32, random_state=1)

    def test_toeplitz(self):
        old_backend = tl.get_backend()
        tl.set_backend("numpy")
        toep_numpy = toeplitz(self.A)
        tl.set_backend("cupy")
        toep_cupy = toeplitz(tl.tensor(self.A))
        toep_cupy = tl.to_numpy(toep_cupy)
        np.testing.assert_allclose(toep_numpy, toep_cupy)
        tl.set_backend(old_backend)

    def test_hoda(self):
        N = 1000
        X = np.random.rand(N, 32, 32)
        y = np.zeros(N)
        y[500:] = 1
        X[y == 0] += 1

        old_backend = tl.get_backend()
        tl.set_backend("numpy")
        means_numpy, Xc_numpy = center(X, y)
        tl.set_backend("cupy")
        means_cupy, Xc_cupy = center(tl.tensor(X), tl.tensor(y))
        Xc_cupy = tl.to_numpy(Xc_cupy)
        means_cupy = tl.to_numpy(means_cupy)
        np.testing.assert_allclose(Xc_numpy, Xc_cupy, atol=1e-6)
        np.testing.assert_allclose(means_numpy, means_cupy, atol=1e-6)
        tl.set_backend(old_backend)


if __name__ == "__main__":
    unittest.main()
