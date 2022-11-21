import math

import ipdb
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
import scipy.linalg
import tensorly as tl
import tensorly.decomposition
import tensorly.tenalg
from sklearn.base import BaseEstimator, TransformerMixin
from tensorly import random as tl_random


class MLSVD(BaseEstimator, TransformerMixin):
    def __init__(self, rank=None):
        self.rank = rank

    def fit(self, X, y=None):
        modes = tuple(range(1, len(X.shape)))
        _, self.factors_ = tensorly.decomposition.partial_tucker(
            X, modes=modes, rank=self.rank
        )
        return self

    def transform(self, X, y=None):
        modes = tuple(range(1, len(X.shape)))
        Xt = tensorly.tenalg.multi_mode_dot(
            X, self.factors_, modes=modes, transpose=True
        )
        return Xt


class HODA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_iter=100,
        tol=1e-13,
        rank=None,
        initialize="identity",
        shrinkage="oas",
        toeplitz=None,
        solver="gevd",
        verbose=False,
        tl_context=None,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.rank = rank
        self.initialize = initialize
        self.shrinkage = shrinkage
        self.toeplitz = toeplitz
        self.solver = solver
        self.verbose = verbose
        self.tl_context = tl_context

    def fit(self, X, y):
        tl_context = self.tl_context
        if self.tl_context is None:
            tl_context = dict()
        X = tl.tensor(X, **tl_context)

        self.classes_, class_counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        shape = X.shape[1:]
        order = len(shape)

        # Initialize projections
        self.projs_ = [None] * order
        for k in range(order):
            if self.initialize == "identity":
                self.projs_[k] = tl.eye(shape[k], self.rank, **tl_context)
            elif self.initialize == "random":
                self.projs_[k] = tl_random.random_tensor(
                    shape=(shape[k], self.rank), **tl_context
                )
                self.projs_[k], _ = tl.qr(self.projs_[k], mode="reduced")
            elif self.initialize == "svd":
                x = tl.unfold(X, k + 1)
                self.projs_[k], _, _ = tl.partial_svd(x, n_eigenvecs=self.rank)
            else:
                raise ValueError("initialize should be one of {identity, random, svd}")

        # Find projections
        self.updates_ = []
        self.scatter_w_ = [None] * order
        self.scatter_b_ = [None] * order

        for self.iter_ in range(self.max_iter):
            if self.verbose:
                print(f"[{self.iter_}/{self.max_iter}]", end="  ")
            new_projs = [None] * order
            for k in range(order):
                X_proj = self._project(X, k)
                X_proj = tl.base.partial_unfold(X_proj, mode=k, skip_begin=1)

                class_means_proj = []
                X_proj_centered = []
                for ci, c in enumerate(self.classes_):
                    where = y == c
                    where = where.reshape((where.shape[0], 1, 1))
                    mean = tl.mean(X_proj, axis=0, where=where)
                    class_means_proj += [mean]
                    X_proj_where = X_proj[y == c]
                    X_proj_centered += [X_proj_where - mean]
                class_means_proj = tl.stack(class_means_proj, axis=0)

                # Calculate whithin class scatter
                scatter_w = 0
                for ci, c in enumerate(self.classes_):
                    scatter_w += tl.tensordot(
                        X_proj_centered[ci],
                        X_proj_centered[ci].conj(),
                        axes=([0, 2], [0, 2]),
                    )
                # Force symmetry
                scatter_w = (scatter_w + scatter_w.conj().T) / 2
                # Force toeplitz
                if self.toeplitz is not None and k in self.toeplitz:
                    scatter_w_toep = [0] * shape[k]
                    for f in range(shape[k]):
                        scatter_w_toep[f] = tl.mean(tl.diag(scatter_w, k=f))
                        # scatter_w_toep[f] *= 1 - (f / (shape[k] - 1))
                    scatter_w = scipy.linalg.toeplitz(scatter_w_toep)
                # Normalize
                # scatter_w /= tl.sum(tl.diag(scatter_w)) / shape[k]
                # Shrinkage regularization
                if self.shrinkage == "oas":
                    shrinkage = oas(scatter_w, X_proj.shape[0])
                else:
                    shrinkage = self.shrinkage[k]
                if self.verbose:
                    print(f"shrinkage[{k}]={shrinkage:.4f}", end="  ")
                mu = tl.sum(tl.diag(scatter_w)) / shape[k]
                scatter_w = (1 - shrinkage) * scatter_w + shrinkage * mu * tl.eye(
                    shape[k], **tl_context
                )
                self.scatter_w_[k] = scatter_w

                # Calculate between class scatter
                class_means_proj -= tl.mean(class_means_proj, axis=0)
                scatter_b = tl.zeros((shape[k], shape[k]))
                for c in range(n_classes):
                    scatter_b += (
                        class_means_proj[c]
                        @ class_means_proj[c].conj().T
                        * class_counts[c]
                    )
                # Force symmetry
                scatter_b = (scatter_b + scatter_b.conj().T) / 2
                # Normalize
                # scatter_b /= tl.sum(tl.diag(scatter_b)) / shape[k]
                self.scatter_b_[k] = scatter_b

                # Solve
                if self.solver == "gevd":
                    scatter_t = scatter_b + scatter_w
                    subset = [shape[k] - self.rank, shape[k] - 1]
                    w, v = scipy.linalg.eigh(
                        scatter_b, scatter_t, subset_by_index=subset
                    )
                elif self.solver == "sr":
                    raise NotImplementedError

                # Orthonormalize for stability
                v, _ = tl.qr(v, mode="reduced")
                new_projs[k] = v

            # Stopping criterion
            break_flag = True
            update = [0] * order
            for k in range(order):
                tol = self.tol * math.prod(self.projs_[k].shape)
                update[k] = tl.norm(new_projs[k] - self.projs_[k])
                self.updates_.append(update)
                if not update[k] < tol:
                    break_flag = False
            self.projs_ = new_projs
            if self.verbose:
                print(f"step={(sum(update)/order):.4e}")
            if break_flag:
                break
        self.updates_ = tl.tensor(self.updates_, **tl_context)
        return self

    def _project(self, X, k):

        order = len(X.shape) - 1
        return tl.tenalg.multi_mode_dot(
            X, self.projs_, modes=range(1, order + 1), skip=k, transpose=True
        )

    def transform(self, X, y=None):
        order = len(X.shape) - 1
        X_trans = tl.tenalg.multi_mode_dot(
            X, self.projs_, modes=range(1, order + 1), transpose=True
        )
        X_trans = X_trans.reshape(X.shape[0], -1)
        return X_trans


def _T(x):
    return jnp.swapaxes(x, -1, -2)


def _H(x):
    return jnp.conj(_T(x))


def symmetrize(x):
    return (x + _H(x)) / 2


def standardize_angle(w, b):
    if jnp.isrealobj(w):
        return w * jnp.sign(w[0, :])
    else:
        # scipy does this: makes imag(b[0] @ w) = 1
        assert not jnp.isrealobj(b)
        bw = b[0] @ w
        factor = bw / jnp.abs(bw)
        w = w / factor[None, :]
        sign = jnp.sign(w.real[0])
        w = w * sign
        return w


@jax.custom_jvp  # jax.scipy.linalg.eigh doesn't support general problem i.e. b not None
def eigh(a, b):
    """
    Compute the solution to the symmetrized generalized eigenvalue problem.

    a_s @ w = b_s @ w @ np.diag(v)

    where a_s = (a + a.H) / 2, b_s = (b + b.H) / 2 are the symmetrized versions of the
    inputs and H is the Hermitian (conjugate transpose) operator.

    For self-adjoint inputs the solution should be consistent with `scipy.linalg.eigh`
    i.e.

    v, w = eigh(a, b)
    v_sp, w_sp = scipy.linalg.eigh(a, b)
    np.testing.assert_allclose(v, v_sp)
    np.testing.assert_allclose(w, standardize_angle(wk_sp))

    Note this currently uses `jax.linalg.eig(jax.linalg.solve(b, a))`, which will be
    slow because there is no GPU implementation of `eig` and it's just a generally
    inefficient way of doing it. Future implementations should wrap cuda primitives.
    This implementation is provided primarily as a means to test `eigh_jvp_rule`.

    Args:
        a: [n, n] float self-adjoint matrix (i.e. conj(transpose(a)) == a)
        b: [n, n] float self-adjoint matrix (i.e. conj(transpose(b)) == b)

    Returns:
        v: eigenvalues of the generalized problem in ascending order.
        w: eigenvectors of the generalized problem, normalized such that
            w.H @ b @ w = I.
    """
    a = symmetrize(a)
    b = symmetrize(b)
    b_inv_a = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(b), a)
    v, w = jax.jit(jax.numpy.linalg.eig, backend="cpu")(b_inv_a)
    v = v.real
    # with loops.Scope() as s:
    #     for _ in s.cond_range(jnp.isrealobj)
    if jnp.isrealobj(a) and jnp.isrealobj(b):
        w = w.real
    # reorder as ascending in w
    order = jnp.argsort(v)
    v = v.take(order, axis=0)
    w = w.take(order, axis=1)
    # renormalize so v.H @ b @ H == 1
    norm2 = jax.vmap(lambda wi: (wi.conj() @ b @ wi).real, in_axes=1)(w)
    norm = jnp.sqrt(norm2)
    w = w / norm
    w = standardize_angle(w, b)
    return v, w


@eigh.defjvp
def eigh_jvp_rule(primals, tangents):
    """
    Derivation based on Boedekker et al.

    https://arxiv.org/pdf/1701.00392.pdf

    Note diagonal entries of Winv dW/dt != 0 as they claim.
    """
    a, b = primals
    da, db = tangents
    if not all(jnp.isrealobj(x) for x in (a, b, da, db)):
        raise NotImplementedError("jvp only implemented for real inputs.")
    da = symmetrize(da)
    db = symmetrize(db)

    v, w = eigh(a, b)

    # compute only the diagonal entries
    dv = jax.vmap(
        lambda vi, wi: -wi.conj() @ db @ wi * vi + wi.conj() @ da @ wi,
        in_axes=(0, 1),
    )(v, w)

    dv = dv.real

    E = v[jnp.newaxis, :] - v[:, jnp.newaxis]

    # diagonal entries: compute as column then put into diagonals
    diags = jnp.diag(-0.5 * jax.vmap(lambda wi: wi.conj() @ db @ wi, in_axes=1)(w))
    # off-diagonals: there will be NANs on the diagonal, but these aren't used
    off_diags = jnp.reciprocal(E) * (_H(w) @ (da @ w - db @ w * v[jnp.newaxis, :]))

    dw = w @ jnp.where(jnp.eye(a.shape[0], dtype=np.bool), diags, off_diags)

    return (v, w), (dv, dw)


def oas(emp_cov, n_samples):
    n_features = emp_cov.shape[0]
    mu = np.trace(emp_cov) / n_features

    # formula from Chen et al.'s **implementation**
    alpha = np.mean(emp_cov**2)
    num = alpha + mu**2
    den = (n_samples + 1.0) * (alpha - (mu**2) / n_features)

    shrinkage = np.real(num / den)
    return max(min(shrinkage, 1), 0)
