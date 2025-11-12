import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import tensorly as tl

try:
    import cupy
    import cupy.linalg
    import cupyx.scipy.linalg
    import cupyx.scipy.sparse.linalg
except ImportError:
    pass


def norm_fro(A):
    return tl.sqrt(tl.sum(A**2))


def toeplitz(A):
    if tl.get_backend() == "numpy":
        return scipy.linalg.toeplitz(A)
    elif tl.get_backend() == "cupy":
        return cupyx.scipy.linalg.toeplitz(A)
    else:
        raise NotImplementedError


def solve_gevdh(
    A,
    B=None,
    solver="lanczos",
    rank=None,
    eigvals_only=False,
    which="LA",
    init=None,
    **solver_params,
):
    if rank is None:
        rank = A.shape[0]

    if solver == "lanczos":
        w, v = _solve_lanczos(A, B=B, **solver_params)
    elif solver == "lobpcg":
        w, v = _solve_lobpcg(A, B=B, rank=rank, init=init, **solver_params)
    elif solver == "svd":
        # Can only be used with symmetric SPD objective
        raise NotImplementedError
    else:
        raise ValueError("(g)evd solver must be on of ['lanczos', 'lobpcg', 'svd']")

    # Truncate
    if which == "LM":
        order = tl.argsort(-tl.abs(w))
    elif which == "LA":
        order = tl.argsort(-w)
    elif which == "SM":
        order = tl.argsort(tl.abs(w))
    elif which == "SA":
        order = tl.argsort(w)
    else:
        raise ValueError("which must be one of ['LM', 'LA', 'SM', 'SA']")
    w = w[order[:rank]]
    v = v[:, order[:rank]]

    # this does not run fast on the GPU
    # min_ev = tl.min(tl.abs(w))
    # max_ev = tl.max(tl.abs(w))
    # if min_ev<tol*max_ev:
    #     null_idc = tl.abs(w)<tol*tl.max(tl.abs(w))
    #     warnings.warn('completing incomplete basis with orthogonal vectors')
    #     basis_completion =  complete_orthonormal_basis(v[:,~null_idc], rank)
    #     v[:,null_idc] = basis_completion
    if eigvals_only:
        return w

    v /= tl.norm(v, axis=0)
    v = flip_signs(v)
    return v, w


def _solve_lanczos(A, B=None, **solver_params):
    if tl.get_backend() == "numpy":
        w, v = scipy.linalg.eigh(A, b=B, **solver_params)
    elif tl.get_backend() == "cupy":
        if B is None:
            w, v = cupy.linalg.eigh(A, **solver_params)
        else:
            import pdb

            pdb.set_trace()
            # L = cupy.linalg.cholesky(B)
            # Y = cupy.linalg.solve(L, A)
            # V = cupy.linalg.solve(L.T, Y)
            # wy, vy = cupy.linalg.eigh(V.T @ A @ V, **solver_params)
            # v = V @ vy
            # w = cupy.empty_like(wy)

            # C = tl.solve(B, A)
            # w, v = cupy.linalg.eigh(C, **solver_params)
            raise NotImplementedError(
                "Lanczos solver for generalized evd is not available for the cupy tensorly backend"
            )
    else:
        raise NotImplementedError
    return w, v


def _solve_lobpcg(A, B=None, rank=None, init=None, **solver_params):
    if init is None:
        init = A
    if rank is None:
        rank = A.shape[1]
    init = init[:, :rank]
    if tl.get_backend() == "numpy":
        w, v, *_ = scipy.sparse.linalg.lobpcg(A, init, B=B, **solver_params)
    elif tl.get_backend() == "cupy":
        w, v, *_ = cupyx.scipy.sparse.linalg.lobpcg(A, init, B=B, **solver_params)
    else:
        raise NotImplementedError
    return w, v


def center(X, y, classes=None):
    _, *shape = X.shape
    order = len(shape)
    if classes is None:
        classes = np.unique(y)
    n_classes = len(classes)

    means = tl.zeros((n_classes, *shape))

    if tl.get_backend() == "cupy":
        # Faster on GPU
        X_centered = tl.zeros((n_classes, *X.shape))
        full_nan = cupy.full_like(X, cupy.nan)
        for ci, c in enumerate(classes):
            where = tl.tensor(y == c)
            where = cupy.expand_dims(where, axis=tuple(np.arange(1, order + 1)))
            X_where = cupy.where(
                where,
                X,
                full_nan,
            )
            means[ci] = cupy.nanmean(X_where, axis=0)
            X_centered[ci] = X_where - means[ci]
        X_centered = cupy.nansum(X_centered, axis=0)
    else:
        X_centered = []
        for ci, c in enumerate(classes):
            X_where = X[y == c]
            means[ci] = tl.mean(X_where, axis=0)
            X_centered.append(X_where - means[ci])
        X_centered = tl.concatenate(X_centered, axis=0)
    if tl.get_backend() == "numpy":
        X_centered = np.zeros_like(X)
        for ci, c in enumerate(classes):
            means[ci] = tl.mean(X[y == c], axis=0)
            X_centered[y == c] = X[y == c] - means[ci]

    # if tl.get_backend() == "cupy":
    #    X_centered = cupy.zeros_like(X)
    # elif tl.get_backend() == "numpy":
    # else:
    #    raise ValueError
    # for ci, c in enumerate(classes):
    #    means[ci] = tl.mean(X[y == c], axis=0)
    #    X_centered[y == c] = X[y == c] - means[ci]

    return means, X_centered


def ridge_regression(X, Y, lambda_=0):
    """
    Compute the ridge regression solution for matrix Y.

    X: Input matrix (n x p)
    Y: Target matrix (n x m)
    lambda_: Regularization parameter

    Returns:
    W: The ridge regression weight matrix (p x m)
    """
    _, p = X.shape
    scatter = X.T @ X
    scale = lambda_ * tl.trace(scatter)
    target = scale * get_eye(p)
    XTX_plus_lambda_I = scatter + target
    XTy = X.T @ Y

    W = tl.solve(XTX_plus_lambda_I, XTy)
    return W


def get_eye(n_features):
    """Cache identity matrix to avoid reallocating it multiple times."""
    if not hasattr(get_eye, "cache"):
        get_eye.cache = {}  # Initialize cache
    key = f"{tl.get_backend}-{n_features}"
    if n_features not in get_eye.cache:
        get_eye.cache[key] = tl.eye(n_features)  # Store once
    return get_eye.cache[key]


def flip_signs(u):
    n_rows, _ = u.shape
    ones = tl.ones(n_rows)
    signs = tl.sign(u.T @ ones)
    return u * signs


def complete_orthonormal_basis(Q, target_dim):
    """
    Given a set of linearly independent vectors in A, complete it to an orthonormal basis
    of dimension `target_dim` using standard basis vectors if necessary.
    """
    Q = Q.copy()
    m, n = Q.shape
    standard_basis = tl.eye(m)

    for i in range(m):
        # Break if the target dimension is reached
        if Q.shape[1] >= target_dim:
            break

        candidate = standard_basis[:, i]  # Pick a standard basis vector

        # Remove components along existing basis
        new_vec = candidate - Q @ (Q.T @ candidate)

        # Normalize and check if it's a valid new basis vector
        norm_new_vec = tl.norm(new_vec)
        if norm_new_vec > 1e-10:
            new_vec /= norm_new_vec  # Normalize
            Q = np.column_stack((Q, new_vec))  # Add to basis
    return Q[:, n:]
