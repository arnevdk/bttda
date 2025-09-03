import math
import warnings

import numpy as np
import scipy.linalg
import tensorly as tl
from numpy.linalg import LinAlgError
import scipy.sparse.linalg
#from hoda.cov import mode_scatter
from scipy.sparse.linalg import ArpackNoConvergence, ArpackError
import scipy.linalg

try:
    import cupy
    import cupyx.scipy.sparse.linalg
    import cupy.linalg
except ImportError:
    pass


def norm_fro(A):
    return tl.sqrt(tl.sum(A**2))


def solve_gevdh(
        A, B=None, solver="lanczos", rank=None, eigvals_only=False, which='LA', init=None, tol=1e-16, **solver_params
):
    if rank is None:
        rank = A.shape[0]

    if solver == "lanczos":
        if tl.get_backend() == 'numpy':
            w,v = scipy.linalg.eigh(A, b=B)
        elif tl.get_backend() == 'cupy':
            C = A
            if B is not None:
                C = tl.solve(B,A)
            w,v = cupy.linalg.eigh(C)
        else:
            raise NotImplementedError

        # Truncate
        if which=='LM':
            idc = tl.argsort(-tl.abs(w))[:rank]
        elif which=='LA':
            w,v = w[-rank:], v[:,-rank:]
        elif which=='SM':
            idc = tl.argsort(tl.abs(w))[:rank]
        elif which=='SA':
            w,v = w[:rank], v[:,:rank]


            #if rank==C.shape[0]:
            #    w,v = cupy.linalg.eigh(C)
            #else:
            #    w,v = cupyx.scipy.sparse.linalg.eigsh(
            #        C, k=rank,
            #        which=which,
            #        return_eigenvectors=~eigvals_only,
            #    )
            #    if np.any(np.isnan(w)):
            #        warnings.warn('cupyx.scipy.sparse.linalg.eigsh failed, using cupy.linalg.eigh')
            #        w,v = cupy.linalg.eigh(C)
    if solver == "svd":
        raise NotImplementedError
    if solver == "lobpcg":
        raise NotImplementedError
    if eigvals_only:
        return w


    # this does not run fast on the GPU
    #min_ev = tl.min(tl.abs(w))
    #max_ev = tl.max(tl.abs(w))
    #if min_ev<tol*max_ev:
    #     null_idc = tl.abs(w)<tol*tl.max(tl.abs(w))
    #     warnings.warn('completing incomplete basis with orthogonal vectors')
    #     basis_completion =  complete_orthonormal_basis(v[:,~null_idc], rank)
    #     v[:,null_idc] = basis_completion
    return v,w


def center(X, y, classes=None):
    _, *shape = X.shape
    order = len(shape)
    if classes is None:
        classes = np.unique(y)
    n_classes = len(classes)

    means = tl.zeros((n_classes, *shape))
    if tl.get_backend() == "cupy":
        X_centered = tl.zeros((n_classes, *X.shape))
        full_nan = cupy.full_like(X, cupy.nan)
        for ci, c in enumerate(classes):
            where = tl.tensor(y==c)
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
        n, p = X.shape
        _, m = Y.shape
        scatter = X.T@X
        scale = lambda_*tl.trace(scatter)
        target = scale*get_eye(p)
        XTX_plus_lambda_I = scatter + target
        XTy = X.T @ Y


        W = tl.solve(XTX_plus_lambda_I, XTy)
        return W

def get_eye(n_features):
    """Cache identity matrix to avoid recomputing it multiple times."""
    if not hasattr(get_eye, "cache"):
        get_eye.cache = {}  # Initialize cache
    if n_features not in get_eye.cache:
        get_eye.cache[n_features] = tl.eye(n_features)  # Store once
    return get_eye.cache[n_features]


def flip_signs(u):
    n_rows, n_cols = u.shape
    ones = tl.ones(n_rows)
    signs = tl.sign(u.T@ones)
    return u * signs
 
def complete_orthonormal_basis(Q, target_dim):
    """
    Given a set of linearly independent vectors in A, complete it to an orthonormal basis
    of dimension 'target_dim' using standard basis vectors if necessary.
    """
    Q = Q.copy()
    m, n = Q.shape


    # Step 2: Add standard basis vectors to complete the basis
    standard_basis = tl.eye(m)  # Standard basis vectors (identity matrix columns)
    
    
    for i in range(m):
        if Q.shape[1] >= target_dim:
            break  # Stop if we've reached the target rank
        
        candidate = standard_basis[:, i]  # Pick a standard basis vector

        # Remove components along existing basis
        new_vec = candidate - Q @ (Q.T @ candidate)

        # Normalize and check if it's a valid new basis vector
        norm_new_vec = tl.norm(new_vec)
        if norm_new_vec > 1e-10:
            new_vec /= norm_new_vec  # Normalize
            Q = np.column_stack((Q, new_vec))  # Add to basis
    return Q[:,n:]


