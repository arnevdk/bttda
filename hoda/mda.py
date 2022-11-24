import autograd.numpy as np
import pymanopt.function
import tensorly as tl
import tensorly.random
from pymanopt import Problem
from pymanopt.manifolds import Product, Stiefel
from pymanopt.optimizers import ConjugateGradient
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import shrunk_covariance
from tensorly.cp_tensor import CPTensor

from hoda import force_toeplitz


class ParafacMDA(BaseEstimator, TransformerMixin):
    def __init__(self, rank=3):
        self.rank = rank

    def fit(self, X, y=None):
        shape = X.shape[1:]
        n_modes = len(shape)
        dtype = X.dtype
        X = tl.tensor(X)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # TODO:  set dtype
        self.proj_ = tensorly.random.random_cp(
            shape,
            self.rank,
            orthogonal=True,
            normalise_factors=True,
        )

        # Whithin class scatter
        self.scatter_w_ = [None] * n_modes
        for k in range(n_modes):
            self.scatter_w_[k] = np.zeros((shape[k], shape[k]), dtype=dtype)
            for c in self.classes_:
                X_cls = X[y == c]
                unfolded = tl.unfold(X_cls, mode=k + 1)
                self.scatter_w_[k] += np.cov(unfolded) / n_classes

        # Between class scatter
        self.scatter_b_ = [None] * n_modes
        cls_means = tl.zeros((n_classes, *shape), dtype=dtype)
        for i, c in enumerate(self.classes_):
            cls_means[i] = tl.mean(X[y == c], axis=0)
        for k in range(n_modes):
            unfolded = tl.unfold(cls_means, mode=k + 1)
            self.scatter_b_[k] = np.cov(unfolded)

        submanifolds = [None] * n_modes
        for k in range(n_modes):
            submanifolds[k] = Stiefel(shape[k], self.rank)
        manifold = Product(submanifolds)

        @pymanopt.function.autograd(manifold)
        def opt_cost(*U_factors):
            U_weights = np.ones(self.rank)
            U_cp = CPTensor((U_weights, U_factors))
            return 1 / self._objective(U_cp)

        problem = Problem(manifold=manifold, cost=opt_cost)
        self.optimizer_ = ConjugateGradient(verbosity=2)
        U_opt = self.optimizer_.run(problem).point
        self.proj_ = CPTensor((np.ones(self.rank), U_opt))
        return self

    def _objective(self, U):
        n_modes = len(U.shape)
        # TODO: replace with multi-mode dot
        num = U
        den = U
        for k in range(n_modes):
            num = num.mode_dot(self.scatter_b_[k], k)
            den = den.mode_dot(self.scatter_w_[k], k)
        num = tl.tenalg.inner(num.to_tensor(), U.to_tensor())
        den = tl.tenalg.inner(den.to_tensor(), U.to_tensor())
        obj = num / den
        print(obj)
        return obj
