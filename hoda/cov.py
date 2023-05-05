import numpy as np


def kron_pca(cov, n_components=2, outer=-1, inner=-1):
    cov_pvl = pitsianis_van_loan_perm(cov, outer, inner)
    pass


def pitsianis_van_loan_perm(cov, outer=-1, inner=-1):
    if outer == inner == -1:
        raise ValueError("Either outer or inner dimension be specified")
    total = cov.shape[0]
    if outer == -1:
        outer = total // inner
    if inner == -1:
        inner = total // outer
    cov_pvl = np.zeros((outer * outer, inner * inner))
    for o1 in range(outer):
        for o2 in range(outer):
            block = cov[o1 * inner : (o1 + 1) * inner, o2 * inner : (o2 + 1) * inner]
            cov_pvl[o1 * outer + o2] = block.flatten()
    return cov_pvl
