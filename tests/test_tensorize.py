import unittest

import ipdb
import numpy as np
from hoda.tensorize import hankel_tensor, hankel_tensor_inv


class TestTensorize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.N = 1

    def test_hankel_even(self):
        raise NotImplementedError

    def test_hankel_odd(self):
        raise NotImplementedError

    def test_hankel_inv_even(self):
        raise NotImplementedError

    def test_hankel_inv_odd(self):
        raise NotImplementedError

    def test_hankel_rec_even(self):
        X = np.random.rand(self.N, 1, 10)
        X_hankel = hankel_tensor(X)
        X_rec = hankel_tensor_inv(X_hankel)
        ipdb.set_trace()
        np.testing.assert_array_equal(X, X_rec)

    def test_hankel_rec_odd(self):
        X = np.random.rand(self.N, 1, 11)
        X_hankel = hankel_tensor(X)
        X_rec = hankel_tensor_inv(X_hankel)
        ipdb.set_trace()
        np.testing.assert_array_equal(X, X_rec)


if __name__ == "__main__":
    unittest.main()
