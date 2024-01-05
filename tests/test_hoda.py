import unittest

import numpy as np
from hoda.hoda import HODA


def generate_order_1(p, samples_per_class, n_classes, independent=False):
    if independent:
        cov = np.eye(p)
    else:
        raise NotImplementedError
    X = []
    y = []
    for ci in range(n_classes):
        mean_c = np.random.rand(p)
        X.append(np.random.multivariate_normal(mean_c, cov, size=samples_per_class))
        y.append(np.ones(samples_per_class) * ci)
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y


class TestHODAOrder1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.X, cls.y = generate_order_1(8, 100000, 4, independent=True)

    def test_independent_tr(self):
        hoda = HODA(max_iter=32, rank=[4], obj="tr", keep_train_info=False)
        hoda.fit(self.X, self.y)
        np.testing.assert_allclose(hoda.scalings_[0], hoda.aps_[0], atol=1e-2, rtol=0)
        np.testing.assert_allclose(
            hoda.cov_[0], np.eye(self.X.shape[-1]), atol=1e-2, rtol=0
        )

    def test_independent_rt(self):
        hoda = HODA(max_iter=32, rank=[4], obj="rt", keep_train_info=False)
        hoda.fit(self.X, self.y)
        np.testing.assert_allclose(hoda.scalings_[0], hoda.aps_[0], atol=1e-2, rtol=0)
        np.testing.assert_allclose(
            hoda.cov_[0], np.eye(self.X.shape[-1]), atol=1e-2, rtol=0
        )
