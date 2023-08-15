import torch


class PointCloud:
    """
    Manifold of point clouds
    """

    def __init__(self, dim, numpoints, base=None, alpha=0.1):
        """

        :param dim: integer d
        :param numpoints: integer n
        :param base: n x d tensor
        :param alpha: float32
        """
        assert numpoints >= dim + 1

        self.d = dim
        self.n = numpoints

        self.manifold_dimension = int(self.d * self.n - self.d * (self.d + 1) / 2)
        self.vert_space_dimension = int(self.d * (self.d + 1) / 2)

        if base is None:
            self.has_base_point = False
        else:
            assert len(base.shape) == 2
            self.has_base_point = True
            self.base_point = self.center_mpoint(base[None, None]).squeeze()

        self.alpha = alpha

    def s_mean(self, x, x0=None, base=None, step_size=1., max_iter=100, tol=1e-3, debug=False):
        """

        :param x: N x M x n x d tensor
        :param x0: N x 1 x n x d tensor
        :return: N x 1 x n x d tensor
        """
        if base is not None:
            if x0 is not None:
                z = x0
                pws_mat = self.s_distance(x, z) ** 2
                error0 = torch.sqrt(torch.sum(pws_mat, 1).min(1).values.max()) + 1e-6
            else:  # not recommended for combinations of large M and n
                pws_mat = self.s_distance(x, x) ** 2
                error0 = torch.sqrt(torch.sum(pws_mat, 1).min(1).values.max()) + 1e-6
                z = x[:, torch.argmin(torch.sum(pws_mat, 1),
                                      1)]  # pick conformation with least distance squared to every other point
            relerror = 1.
            k = 1
            while relerror > tol and k <= max_iter:
                # compute grad
                grad_Wz = - torch.mean(self.s_log(z, x), 2)
                z = z - step_size * grad_Wz
                error = self.norm(z, grad_Wz[:, :, None]).max()
                relerror = error / error0
                if debug:
                    print(f"{k} | relerror = {relerror}")

                k = k + 1

            return self.align_mpoint(z, base=base)
        else:
            return self.s_mean(x, x0=x0, base=self.base_point, step_size=step_size, max_iter=max_iter, tol=tol, debug=debug)

    def s_geodesic(self, x, y, tau, base=None, step_size=1., max_iter=100, tol=1e-3, debug=False):
        """

        :param x: N x 1 x n x d tensor
        :param y: N x 1 x n x d tensor
        :param tau: M tensor
        :return: N x M x n x d tensor
        """
        if base is not None:
            assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1] == 1

            error0 = self.s_distance(x, y).max() + 1e-6
            relerror = 1.
            k = 1
            z = torch.ones(len(tau))[None, :, None, None] * y
            while relerror > tol and k <= max_iter:
                # compute grad  
                grad_Wzx = - self.s_log(z, x)[:, :, 0]
                grad_Wzy = - self.s_log(z, y)[:, :, 0]
                grad_Wz = (1 - tau[None, :, None, None]) * grad_Wzx + tau[None, :, None, None] * grad_Wzy

                # update z
                z = z - step_size * grad_Wz

                # compute new error
                error = self.norm(z, grad_Wz[:, None]).max()
                relerror = error / error0
                if debug:
                    print(f"{k} | relerror = {relerror}")

                k = k + 1

            return self.align_mpoint(z, base=base)
        else:
            return self.s_geodesic(x, y, tau, base=self.base_point, step_size=step_size, max_iter=max_iter, tol=tol,
                                   debug=debug)

    def s_bezier(self, x, tau, base=None, step_size=1., max_iter=100, tol=1e-4, debug=False):
        """

        :param x: N x 1 x n x d tensor
        :param tau: 1 tensor
        :return: 1 x 1 x n x d tensor
        """
        N = x.shape[0]
        if N == 2:
            if base is not None:
                return self.s_geodesic(x[0][None], x[1][None], tau, base=base,
                                       step_size=step_size, max_iter=max_iter, tol=tol, debug=debug)
            else:
                return self.s_geodesic(x[0][None], x[1][None], tau, base=self.base_point,
                                       step_size=step_size, max_iter=max_iter, tol=tol, debug=debug)
        else:
            return self.s_geodesic(
                self.s_bezier(x[0:N - 1], tau, base=base, step_size=step_size, max_iter=max_iter, tol=tol, debug=debug),
                self.s_bezier(x[1:], tau, base=base, step_size=step_size, max_iter=max_iter, tol=tol, debug=debug),
                tau, base=base, step_size=step_size, max_iter=max_iter, tol=tol, debug=debug)

    def s_exp(self, x, X, c=1 / 4, base=None, step_size=1., max_iter=100, tol=1e-3, debug=False):
        """

        :param x: N x 1 x n x d tensor
        :param X: N x 1 x n x d tensor
        :return: N x 1 x n x d tensor
        """
        if base is not None:
            K = int(c * self.norm(x, X[:, :, None]).max()) + 1
            print(f"computing exp in {K} steps")
            x0 = x
            x1 = x + 1 / K * X

            k = 1
            xkk = x0
            xk = x1
            while k < K:
                x_new = self.s_geodesic(xkk, xk, torch.tensor([2.]), step_size=step_size, max_iter=max_iter, tol=tol,
                                        debug=debug)
                xkk = xk
                xk = x_new
                k = k + 1

            return self.align_mpoint(xk, base=base)
        else:
            return self.s_exp(x, X, c=c, base=self.base_point, step_size=step_size, max_iter=max_iter, tol=tol,
                              debug=debug)

    def s_log(self, x, y, asvector=False):
        """

        :param x: N x M x n x d tensor
        :param y: N x M' x n x d tensor
        :param asvector:
        :return: N x M x M' x n x d tensor
        """
        assert x.shape[0] == y.shape[0]
        N = x.shape[0]
        M = x.shape[1]
        MM = y.shape[1]

        prelog = self.s_prelog(x, y, asvector=True)
        H = self.metric_tensor(x, asmatrix=True)
        L, Q = torch.linalg.eigh(H)

        vertical_dim = self.vert_space_dimension
        log = torch.einsum("NMxy,NMy,NMzy,NMLz->NMLx",
                           Q[:, :, :, vertical_dim:], 1/L[:, :, vertical_dim:], Q[:, :, :, vertical_dim:], prelog)

        # log = torch.zeros((N, M, MM, self.n * self.d))
        # for m in range(M):
        #     for mm in range(MM):
        #         log[:, m, mm] = torch.linalg.lstsq(H[:, m], prelog[:, m, mm]).solution

        if asvector:
            return log
        else:
            return log.reshape(N, M, MM, self.n, self.d)

    def s_prelog(self, x, y, asvector=False):
        """

        :param x: N x M x n x d tensor
        :param y: N x M' x n x d tensor
        :param asvector:
        :return: N x M x M' x n x d tensor
        """
        assert x.shape[0] == y.shape[0]
        N = x.shape[0]
        M = x.shape[1]
        MM = y.shape[1]

        x_pairwise_distances = self.pairwise_distances(x)
        x_pairwise_distances += torch.eye(self.n)  # for numerical stability in log

        y_pairwise_distances = self.pairwise_distances(y)
        y_pairwise_distances += torch.eye(self.n)  # for numerical stability in log

        predists = 1 / 2 * torch.log(x_pairwise_distances[:, :, None, :, :] / y_pairwise_distances[:, None, :, :, :])

        xixj = x[:, :, None, :, None] - x[:, :, None, None, :]

        prelogs = - predists[:, :, :, :, :, None] \
                  * xixj / x_pairwise_distances[:, :, None, :, :, None]

        prelog = torch.sum(prelogs, 4)

        # alpha * correction term
        x_gyration = self.gyration_matrix(x)
        y_gyration = self.gyration_matrix(y)

        precorrections = torch.log(torch.det(x_gyration[:, :, None, :, :]) / torch.det(y_gyration[:, None, :, :, :]))

        xc = self.center_mpoint(x)
        L, Q = torch.linalg.eigh(x_gyration)
        gxi = torch.einsum("NMab,NMb,NMcb,NMic->NMia", Q, 1 / L, Q, xc)

        prelogcorrections = - 2 * gxi[:, :, None] * precorrections[:, :, :, None, None]

        if asvector:
            return (prelog + self.alpha * prelogcorrections).reshape(N, M, MM, self.n * self.d)
        else:
            return prelog + self.alpha * prelogcorrections

    def norm(self, x, X):
        """

        :param x: N x M x n x d tensor
        :param X: N x M x L x n x d tensor
        :return: N x M x L tensor
        """
        assert x.shape[0] == X.shape[0]

        N = x.shape[0]
        M = x.shape[1]
        L = X.shape[2]

        norm = torch.zeros(N, M, L)
        for l in range(L):
            norm[:, :, l] = torch.sqrt(
                self.inner(x, X[:, :, l, :, :][:, :, None, :, :], X[:, :, l, :, :][:, :, None, :, :])[:, :, 0, 0])

        return norm

    def s_distance(self, x, y):
        """
        Manifold distance between points x and y
        :param x: N x M x n x d tensor
        :param y: N x M' x n x d tensor
        :return: N x M x M' tensor
        """
        assert x.shape[0] == y.shape[0]

        x_pairwise_distances = self.pairwise_distances(x)
        x_pairwise_distances += torch.eye(self.n)  # for numerical stability in log

        y_pairwise_distances = self.pairwise_distances(y)
        y_pairwise_distances += torch.eye(self.n)  # for numerical stability in log

        predists = (1 / 2 * torch.log(
            x_pairwise_distances[:, :, None, :, :] / y_pairwise_distances[:, None, :, :, :])) ** 2

        # alpha * correction term
        x_gyration = self.gyration_matrix(x)
        y_gyration = self.gyration_matrix(y)

        corrections = torch.log(torch.det(x_gyration[:, :, None, :, :]) / torch.det(y_gyration[:, None, :, :, :])) ** 2

        return torch.sqrt(1 / 2 * torch.sum(predists, [3,
                                                       4]) + self.alpha * corrections)  # factor 1/2 in first term because we count everything double

    def inner(self, x, X, Y):
        """

        :param x: N x M x n x d tensor
        :param X: N x M x L x n x d tensor
        :param Y: N x M x K x n x d tensor
        :return: N x M x L x K tensor
        """
        assert x.shape[0] == X.shape[0] == Y.shape[0]

        H = self.metric_tensor(x)
        inner = torch.einsum("NMijab,NMLia,NMKjb->NMLK", H, X, Y)

        return inner

    def pairwise_distances(self, x):
        """
        Function that computes the pairwise distance matrix of a point cloud x
        :param x: N x M x n x d tensor
        :return: N x M x n x n tensor with values [(\|x_i - x_j\|^2)_ij]_NM
        """
        x_gram_mat = torch.einsum("NMia,NMja->NMij", x, x)

        x_gram_diag = torch.diagonal(x_gram_mat, dim1=2, dim2=3)
        x_gram_diag_mat = torch.einsum("NMi,j->NMij", x_gram_diag, torch.ones((self.n,)))

        return x_gram_diag_mat - 2 * x_gram_mat + torch.transpose(x_gram_diag_mat, 2, 3)

    def gyration_matrix(self, x):
        """

        :param x: N x M x n x d
        :return: N x M x d x d
        """
        xc = self.center_mpoint(x)
        return torch.einsum("NMia,NMib->NMab", xc, xc)

    def metric_tensor(self, x, asmatrix=False):
        """

        :param x: N x M x n x d tensor
        :param asmatrix:
        :return: N x M x n x n x d x d tensor or N x M x nd x nd tensor if asmatrix==True
        """
        N = x.shape[0]
        M = x.shape[1]

        x_pairwise_distances = self.pairwise_distances(x)
        x_pairwise_distances += torch.eye(self.n)  # for numerical stability in log

        xixj = x[:, :, :, None] - x[:, :, None, :]
        xij = torch.einsum("NMija,NMijb->NMijab", xixj, xixj)
        A = - xij \
            / x_pairwise_distances[:, :, :, :, None, None] ** 2
        # fix diagonal
        Adiag = - torch.sum(A, 3).permute(0, 1, 3, 4, 2)
        A += torch.diag_embed(Adiag).permute(0, 1, 4, 5, 2, 3)

        # alpha * correction term
        xc = self.center_mpoint(x)
        x_gyration = self.gyration_matrix(x)

        L, Q = torch.linalg.eigh(x_gyration)
        yi = torch.einsum("NMab,NMb,NMcb,NMic->NMia", Q, 1 / L, Q, xc)

        B = 4 * torch.einsum("NMia,NMjb->NMijab", yi, yi)

        if asmatrix:
            return (A + self.alpha * B).permute(0, 1, 2, 5, 3, 4).reshape(N, M, self.n * self.d, self.n * self.d)
        else:
            return A + self.alpha * B

    def orthonormal_basis(self, x, asvector=False):
        """

        :param x: N x M x n x d tensor
        :param asvector:
        :return: N x M x L x n x d tensor with L:= manifold_dimension
        """
        N = x.shape[0]
        M = x.shape[1]

        H = self.metric_tensor(x, asmatrix=True)
        L, Q = torch.linalg.eigh(H)

        vertical_dim = self.vert_space_dimension
        horizontal_vectors = Q.permute(0, 1, 3, 2)[:, :, vertical_dim:]
        rescaling_factors = 1 / torch.sqrt(L[:, :, vertical_dim:])
        horizontal_vectors = rescaling_factors[:, :, :, None] * horizontal_vectors
        if asvector:
            return horizontal_vectors
        else:
            return horizontal_vectors.reshape(N, M, self.manifold_dimension, self.n, self.d)

    def coordinates_in_basis(self, x, X, Xi):
        """
        compute coefficients c of X in basis \Xi, i.e., X = c^i \Xi_i
        :param x: N x M x n x d tensor
        :param X: N x M x n x d tensor
        :param Xi: N x M x L x n x d
        :return: N x M x L tensor with L:= manifold_dimension
        """
        return self.inner(x, X[:, :, None], Xi)[:, :, 0]

    def tvector_in_basis(self, c, Xi):
        """
        compute tvector X from coordinates c in basis \Xi, i.e., X = c^i \Xi_i
        :param c: N x M x L tensor with L:= manifold_dimension
        :param Xi: N x M x L x n x d tensor
        :return: N x M x n x d tensor
        """
        return torch.einsum("NML,NMLia->NMia", c, Xi)

    def align_mpoint(self, x, base=None):
        """

        :param x: N x M x n x d tensor
        :param base: n x d tensor
        :return: N x M x n x d tensor
        """
        if base is not None:
            base_ = self.center_mpoint(base[None, None]).squeeze()
            xc = self.center_mpoint(x)
            O = self.least_orthogonal(xc, base=base_)
            return self.orthogonal_transform_mpoint(xc, O)
        else:
            assert self.base_point is not None
            return self.align_mpoint(x, base=self.base_point)

    def center_mpoint(self, x):
        """

        :param x: N x M x n x d tensor
        :return: N x M x n x d tensor
        """
        t = torch.mean(x, 2)
        return self.translate_mpoint(x, -t)

    def least_orthogonal(self, x, base=None):
        """
        Solve inf_O \sum_i \|y_i - Ox_i\|^2, i.e., how to rotate x such that it's closest to y:=base
        :param x: N x M x n x d tensor
        :param base: n x d tensor
        :return: N x M x d x d tensor
        """
        if base is not None:
            inertia_tensor = torch.einsum("NMia,ib->NMab", x, base) / self.n
            svd = torch.svd(inertia_tensor)

            O = torch.einsum("NMcb,NMab->NMca", svd.V, svd.U)
            return O
        else:
            assert self.base_point is not None
            return self.least_orthogonal(x, base=self.base_point)

    def orthogonal_transform_mpoint(self, x, O):
        """
        Compute O . (x_1, ..., x_n) := (O x_1, ..., O x_n)
        :param x: N x M x n x d tensor
        :param O: N x M x d x d tensor
        :return:
        """
        return torch.einsum("NMba,NMia->NMib", O, x)

    def translate_mpoint(self, x, t):
        """
        Compute t . (x_1, ..., x_n) := (x_1 - t, ..., x_n - t)
        :param x: N x M x n x d tensor
        :param t: N x M x d tensor
        :return: N x M x n x d tensor
        """
        return x + t[:, :, None, :]

    def horizontal_projection_tvector(self, x, X):
        """

        :param x: N x M x n x d tensor
        :param X: N x M x M' x n x d tensor
        :return: N x M x M' x n x d tensor
        """
        assert x.shape[0] == X.shape[0] and x.shape[1] == X.shape[1]
        N = x.shape[0]
        M = x.shape[1]

        xc = self.center_mpoint(x)
        x_gyration = self.gyration_matrix(x)
        L, Q = torch.linalg.eigh(x_gyration)

        vertical_basis = torch.zeros((N, M, self.vert_space_dimension, self.n, self.d))

        for i in range(self.d):
            ei = torch.zeros(self.d)
            ei[i] = 1.
            vi = ei[None] * torch.ones(N, M, self.n)[:, :, :, None]
            vertical_basis[:, :, i] = 1 / self.n ** (1 / 2) * vi
            for j in range(self.d):
                if j > i:
                    Gij = torch.zeros((self.d, self.d))
                    Gij[i, j] = 1
                    Gij[j, i] = -1
                    Ld = torch.einsum('ab,NMa->NMab', torch.eye(self.d), L ** (-1 / 2))
                    QLGijLQt = Q @ Ld @ Gij[None, None] @ Ld @ Q.transpose(2, 3)
                    normalisation = torch.sqrt((L[:, :, i] * L[:, :, j]) / (L[:, :, i] + L[:, :, j]))[:, :, None, None]
                    ind = self.d + int((self.d * (self.d - 1) / 2) - (self.d - i) * ((self.d - i) - 1) / 2 + j - i - 1)
                    vij = torch.einsum("NMab,NMib->NMia", normalisation * QLGijLQt, xc)
                    vertical_basis[:, :, ind] = vij

                    # project X onto vertical space
        VX_inner = torch.einsum("NMVia,NMLia->NMVL", vertical_basis, X)
        Vproj_X = torch.einsum("NMVL,NMVia->NMLia", VX_inner, vertical_basis)

        return X - Vproj_X
