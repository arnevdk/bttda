# g = copy(Xt_centered)
# for k in range(order):
#    modes = [kk + 1 for kk in range(k + 1, order)]
#    x = tl.tenalg.multi_mode_dot(
#        X_centered, self.weights_[k + 1 :], transpose=True, modes=modes
#    )
#    cov_x, shrink_x = mode_scatter(
#        x,
#        k,
#        # shrinkage=self.shrinkage,
#        assume_centered=True,
#        # toeplitz=self.toeplitz,
#        # taper=self.taper,
#    )
#    #cov_x /= math.prod(x.shape) / x.shape[k + 1] - 1
#    print(tl.trace(cov_x))
#    cov_g, shrink_g = mode_scatter(
#        g,
#        k,
#        # shrinkage=self.shrinkage,
#        assume_centered=True,
#    )
#    #cov_g /= math.prod(g.shape) / g.shape[k + 1] - 1
#    P_inv = tl.diag(1 / tl.diag(cov_g))
#    self.aps_[k] = cov_x @ (
#        tl.solve(cov_g @ P_inv, self.weights_[k].T).T @ P_inv
#    )


#    g = tl.tenalg.mode_dot(g, self.aps_[k], k + 1)
# g = tl.tenalg.mode_dot(X_centered, self.weights_[k].T, k + 1)

# for k in range(order):
#    cov_x, shrink_x = mode_scatter(
#        X_centered,
#        k,
#        # shrinkage=self.shrinkage,
#        shrinkage=0,
#        assume_centered=True,
#        # toeplitz=self.toeplitz,
#        # taper=self.taper,
#    )
#    cov_x /= n_samples * math.prod(shape) / shape[k] - 1

#    g = tl.tenalg.mode_dot(X_centered, self.weights_[k].T, k + 1)
#    cov_g, shrink_g = mode_scatter(
#        g,
#        k,
#        # shrinkage=self.shrinkage,
#        shrinkage=0,
#        assume_centered=True,
#    )
#    cov_g /= n_samples * math.prod(shape) / shape[k] - 1
#    # Preconditioner
#    P_inv = tl.diag(1 / tl.diag(cov_g))
#    self.aps_[k] = cov_x @ (
#        tl.solve(cov_g @ P_inv, self.weights_[k].T).T @ P_inv
#    )

# cov_X = KroneckerCovariance(estimator="mle", assume_centered=True)
# cov_X.fit(X_centered, X_centered)
# cov_G = KroneckerCovariance(estimator="mle", assume_centered=True)
# cov_G.fit(Xt_centered, Xt_centered)

# for k in range(order):
#    cov_x, shrink_x = mode_scatter(
#        X_centered,
#        k,
#        assume_centered=True,
#        shrinkage=0,
#    )
#    cov_x /= n_samples * math.prod(shape) / shape[k] - 1

#    # cov_g, shrink_g = mode_scatter(
#    #    Xt_centered,
#    #    k,
#    #    assume_centered=True,
#    #    shrinkage=0,
#    # )
#    # cov_g /= n_samples * math.prod(self.rank_) / self.rank_[k] - 1
#    # cov_x = cov_X.covs_[k]
#    cov_g = cov_G.covs_[k]
#    if k:
#        cov_x /= tl.trace(cov_x) / shape[k]
#        cov_x *= tl.trace(cov_g) / self.rank_[k]

#    P_inv = tl.diag(1 / tl.diag(cov_g))
#    self.aps_[k] = cov_x @ (
#        tl.solve(cov_g @ P_inv, self.weights_[k].T).T @ P_inv
#    )

# L = order
# dims = (*self.rank_, *shape)
# hopls = TOT(
#    dims=dims,
#    L=L,
#    M=2,
#    N=30,
#    R=shape,
#    mu1=5.5e-3,
#    mu2=1e-3,
#    mu3=1e-4,
#    tol=1e-4,
#    max_itr=100,
#    density=0.2,
#    # x=tl.tensor(np.random.rand(n_samples, *dims[:L])),
#    # y=tl.tensor(np.random.rand(n_samples, *dims[L:])),
#    x=Xt_centered,
#    y=X_centered,
# )
# res = hopls.fit(verbose=True)
# self.aps_ = [tl.tensor(a) for a in res[order:]]

# n_samples = X_centered.shape[0]
# X_vec = X_centered.reshape((n_samples, -1))
# G_vec = Xt_centered.reshape((n_samples, -1))
# cov_X = X_vec.T @ X_vec / (n_samples - 1)
# cov_G = G_vec.T @ G_vec / (n_samples - 1)
# self.A_ = cov_X @ (tl.solve(cov_G, tl.tenalg.kronecker(self.weights_).T).T)
# self.A_ = self.A_.reshape((*shape, *self.rank_))

# self.aps_ = [w.copy() / tl.norm(w) for w in self.weights_]


# x = tl.unfold(X, k + 1)
# modes = range(1, order + 1)
# g = tl.tenalg.multi_mode_dot(Xt, self.aps_, modes=modes, skip=k)
# g = tl.unfold(g, k + 1)
# ap, *_ = cupy.linalg.lstsq(g.T, x.T)
# ap = ap.T
# print(tl.metrics.regression.MSE(ap, self.aps_[k]))
# self.aps_[k] = ap

#    def _init_forward(self, X, Xt, X_centered, Xt_centered, y):
#        self.aps_ = [copy(w) for w in self.weights_]
# n_samples, *shape = X.shape
# order = len(shape)

## cov_G = KroneckerCovariance(estimator="mle")
## cov_G.fit(Xt_centered, Xt_centered)

# self.aps_ = [None] * order

# for k in range(order):
#    modes = range(1, order + 1)
#    X_centered_proj = tl.tenalg.multi_mode_dot(
#        X_centered,
#        self.weights_,
#        modes=modes,
#        skip=k,
#        transpose=True,
#    )
#    cov_x, _ = mode_scatter(
#        X_centered_proj,
#        k,
#        assume_centered=True,
#        shrinkage=self.shrinkage,
#        # shrinkage=0,
#        # toeplitz=self.toeplitz,
#        # taper=self.taper,
#    )
#    cov_x /= n_samples * math.prod(self.rank_) / self.rank_[k] - 1

#    cov_g, shrink_g = mode_scatter(
#        Xt_centered,
#        k,
#        assume_centered=True,
#        shrinkage=0,
#    )
#    cov_g /= n_samples * math.prod(self.rank_) / self.rank_[k] - 1
#    # cov_g = cov_G.covs_[k]

#    P_inv = tl.diag(1 / tl.diag(cov_g))
#    self.aps_[k] = cov_x @ (
#        tl.solve(cov_g @ P_inv, self.weights_[k].T).T @ P_inv
#    )

# def _fit_forward(self, X, Xt, X_centered, Xt_centered, y):
#    """
#    https://etna.math.kent.edu/vol.55.2022/pp92-111.dir/pp92-111.pdf
#    HOPLS
#    """

# def _fit_forward(self, X, Xt, X_centered, Xt_centered, y):
#    n_samples, *shape = X.shape
#    order = len(shape)

#    self.cov_x_ = [None] * order
#    self.cov_g_ = [None] * order
#    self.scale_x_ = tl.zeros(order)
#    self.scale_g_ = tl.zeros(order)
#    self.aps_ = [None] * order
#    for k in range(order):
#        cov_x, shrink_x = mode_scatter(
#            X_centered,
#            k,
#            assume_centered=True,
#            # shrinkage=self.shrinkage,
#            # toeplitz=self.toeplitz,
#            # taper=self.taper,
#        )
#        cov_g, shrink_g = mode_scatter(
#            Xt_centered,
#            k,
#            # shrinkage=self.shrinkage,
#            assume_centered=True,
#        )

#        scale_x = (tl.trace(cov_x) / shape[k]) / (
#            n_samples * math.prod(shape) / shape[k] - 1
#        )
#        scale_g = (tl.trace(cov_g) / self.rank_[k]) / (
#            n_samples * math.prod(self.rank_) / self.rank_[k] - 1
#        )

#        cov_x /= tl.trace(cov_x) / shape[k]
#        cov_g /= tl.trace(cov_g) / self.rank_[k]

#        self.cov_g_[k] = cov_g
#        self.cov_x_[k] = cov_x
#        self.scale_x_[k] = scale_x
#        self.scale_g_[k] = scale_g
#        # self.aps_[k] = cov_x @ tl.solve(cov_g, self.weights_[k].T).T
#        P_inv = tl.diag(1 / tl.diag(cov_g))
#        self.aps_[k] = cov_x @ (
#            tl.solve(cov_g @ P_inv, self.weights_[k].T).T @ P_inv
#        )

#    self.scale_x_ = tl.mean(self.scale_x_)
#    self.scale_g_ = tl.mean(self.scale_g_)

# def _fit_forward(self, X, Xt, X_centered, Xt_centered, y, _):
#    n_samples, *shape = X.shape
#    order = len(shape)
#    x_proj_part = [None] * order
#    for k in range(order):
#        modes = range(1, order + 1)
#        x_proj_part[k] = tl.tenalg.multi_mode_dot(
#            X_centered, self.weights_, modes=modes, skip=k, transpose=True
#        )

#    iterator = range(1, self.max_iter + 1)
#    if self.verbose:
#        iterator = tqdm(iterator)
#        iterator.set_description("Forward model ")
#    converged = False
#    for self.iter_ in iterator:
#        if self.extra_train_info:
#            X_approx = self.inv_transform(Xt)
#            err = X - X_approx

#            n_samples = X.shape[0]
#            G_flat = tl.to_numpy(
#                Xt.reshape((n_samples, -1), order="F"),
#            )
#            err_flat = tl.to_numpy(err.reshape((n_samples, -1), order="F"))
#            cross_corr = np.corrcoef(G_flat, err_flat, rowvar=False)
#            cross_corr = cross_corr[G_flat.shape[-1] :, : G_flat.shape[-1]].T
#            print(
#                f"MSE: {tl.metrics.regression.MSE(X, X_approx)} corr: {tl.metrics.regression.MSE(cross_corr, 0)}"
#            )

#        if converged:
#            break

#        converged = True

#        for k in range(order):
#            modes = range(1, order + 1)
#            x = tl.tenalg.multi_mode_dot(
#                x_proj_part[k], self.aps_, modes=modes, skip=k
#            )
#            cov_x, _ = mode_scatter(
#                x,
#                k,
#                assume_centered=True,
#                # shrinkage=self.shrinkage,
#                # toeplitz=self.toeplitz,
#                # taper=self.taper,
#                shrinkage=0,
#            )
#            cov_x /= n_samples * math.prod(shape) / shape[k] - 1

#            g = tl.tenalg.multi_mode_dot(
#                Xt_centered, self.aps_, modes=modes, skip=k
#            )
#            cov_g, shrink_g = mode_scatter(
#                g,
#                k,
#                assume_centered=True,
#                # shrinkage=self.shrinkage,
#                shrinkage=0,
#            )
#            cov_g /= n_samples * math.prod(shape) / shape[k] - 1
#            P_inv = tl.diag(1 / tl.diag(cov_g))
#            ap = cov_x @ (tl.solve(cov_g @ P_inv, self.weights_[k].T).T @ P_inv)

#            # ap = cov_x @ (tl.solve(cov_g, self.weights_[k].T).T)
#            update = tl.metrics.regression.MSE(
#                ap / tl.norm(self.aps_[k]), self.aps_[k] / tl.norm(self.aps_[k])
#            )
#            self.aps_[k] = ap

#        converged = update < self.tol and converged
