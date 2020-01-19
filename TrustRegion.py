import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


class TrustRegion:

    def __init__(self, x0=np.array([0.7, -3.3]), delta_max=50.0, delta_0=0.5, p_eps=1e-6, yita=0.15, max_iter=20):
        assert(delta_max > 0)
        assert(0.0 <= delta_0 < delta_max)
        assert(0.0 <= yita < 0.25)
        self.x0 = x0  # (2, 1)
        self.delta_max = delta_max
        self.delta_0 = delta_0
        self.p_eps = p_eps
        self.yita = yita
        self.max_iter = max_iter
        self.hist = None

    @staticmethod
    def f(x_k):
        x1_, x2_ = x_k[0], x_k[1]
        return -10.0 * x1_ ** 2 + 10.0 * x2_ ** 2 + 4.0 * np.sin(x1_ * x2_) - 2.0 * x1_ + x1_ ** 4

    @staticmethod
    def nabla_f(x_k):  # Gradient of f
        x1_, x2_ = x_k[0], x_k[1]
        partial_x1 = -20.0 * x1_ + 4.0 * x2_ * np.cos(x1_ * x2_) - 2 + 4.0 * x1_ ** 3
        partial_x2 = 20.0 * x2_ + 4.0 * x1_ * np.cos(x1_ * x2_)
        return np.array((partial_x1, partial_x2))  # (2, 1)

    @staticmethod
    def hassian_matrix(x_k):  # Hassian Matrix of f
        x1_, x2_ = x_k[0], x_k[1]
        partial_11 = -20.0 - 4.0 * x2_ ** 2 * np.sin(x1_ * x2_)
        partial_12 = 4.0 * np.cos(x1_ * x2_) - 4 * x1_ * x2_ * np.sin(x1_ * x2_)
        partial_21 = partial_12
        partial_22 = 20.0 - 4.0 * x1_ * x1_ * np.sin(x1_ * x2_)
        return np.array(((partial_11, partial_12),
                         (partial_21, partial_22)))

    def calc_p(self, x_k):
        # min 1/2 * x^T * A * x + b^T * x + c.
        # Without constraints, the solution will be far away from the initial point.
        # This method is not used.
        mat_a_ = self.hassian_matrix(x_k)
        b_ = self.nabla_f(x_k)
        return np.linalg.solve(mat_a_, -b_)

    @staticmethod
    def zero_point(f, x0_=1.0, eps_=1e-2, max_iter_=500):
        # use fixed point to solve f(x) = 0. That is, find x such that f(x) + x = x.
        # here is a rough solution. When eps_ is too small, it might converge at half.
        # As m_k is an approximation for f(x).# As m_k is an approximation for f(x).
        # When eps_ = 1e-2, the solution will not always satisfy the constraints strictly.
        def ff(x):
            return f(x) + x

        x_, ffx_ = x0_, ff(x0_)
        i = 0
        while i < max_iter_:
            x_ = ffx_
            ffx_ = ff(x_)
            if abs(ffx_ - x_) <= eps_ or abs((ffx_ - x_) / x_) < eps_:
                break
            i += 1
        if i >= max_iter_:  # The fix point method doesn't converge
            sys.stderr.write("Warning: Zero Point Not Found. x = %f, f(x) + x = %f \n" % (x_, ffx_))
        return x_

    def calc_p_constrain(self, x_k, delta_):
        # solve p.
        # min f(x) = 1/2 * x^T * A * x + b^T * x + c
        # s.t g(x) = 1/2 * x^T * x - h <= 0. h = 0.5 * delta^2
        # KKT condition:
        # L(x, \lambda) = f(x) + \lambda g(x)
        # \frac{\partial L}{\partial x} = 0 ==> x = -(A+\lambda I)^{-1} * b
        # g(x) <= 0
        # \lambda x * g(x) = 0
        # \lambda >= 0
        h_ = 0.5 * delta_ ** 2
        mat_a_ = self.hassian_matrix(x_k)
        b_ = self.nabla_f(x_k)
        # Consider \lambda = 0
        p_ = np.linalg.solve(mat_a_, -b_)  # It's faster than solving inverse matrix.

        def g_(x_):
            return 0.5 * np.dot(x_, x_) - h_

        if g_(p_) <= 0:
            return p_

        # Consider \lambda > 0. That is, g(x) = 0.

        def get_x_(lambda_):
            return np.linalg.solve((mat_a_ + lambda_ * np.identity(mat_a_.shape[0])), -b_)

        def g_lambda(lambda_):
            x_ = get_x_(lambda_)
            return g_(x_)

        lambda_res = self.zero_point(g_lambda)
        sys.stderr.write("lambda = %f\n" % lambda_res)
        p_ = get_x_(lambda_res)
        sys.stderr.write("g(p) = %f\n" % g_(p_))
        return p_

    def m_k(self, x_k, p):  # Second order Taylor expansion of function f.
        f_k = self.f(x_k)
        nabla_f_k = self.nabla_f(x_k)
        hassian_mat_k = self.hassian_matrix(x_k)
        a = np.dot(nabla_f_k, p)
        b = 0.5 * np.matmul(np.matmul(p, hassian_mat_k), p.transpose())
        return f_k + a + b

    def rho_k(self, x_k, p):  # calculate rho
        df_k = self.f(x_k) - self.f(x_k + p)  # actual reduction
        dm_k = self.f(x_k) - self.m_k(x_k, p)  # predicted reduction
        return df_k / dm_k

    def calc(self):
        self.hist = pd.DataFrame(columns=['x', 'p', 'delta', 'rho', 'f'])
        delta, x, p, rho = self.delta_0, self.x0, 0.0, 0.0
        # delta: search radius
        # x: solution point at iteration k
        # p: x_{k+1} = x_{k} + p
        # if rho <= 0, this step causes a larger value of f. We should reduce scope of Trust Region.
        # if rho <= 0.25, the next solution is not good. We should reduce scope of Trust Region.
        # if 0.25 < rho < 0.75, the next solution is not bad. We should maintain Trust Region.
        # if rho >= 0.75, the next solution is good. In this situation,
        #     if the next solution reaches the boundary of Trust Region,
        #     we should expand the scope of Trust Region.
        for k in range(self.max_iter):
            sys.stderr.write("Trust Region Iteration %d===================\n" % k)
            p = self.calc_p_constrain(x, delta)
            lenp = np.linalg.norm(p)
            if lenp < self.p_eps:
                break
            rho = self.rho_k(x, p)
            fx = self.f(x)
            self.hist = self.hist.append({"x": x, "p": p, "delta": delta, "rho": rho, "f": fx}, ignore_index=True)
            sys.stderr.write("x:" + str(x) + "\np:" + str(p) + "\ndelta:" + str(delta)
                             + "\nrho:" + str(rho) + "\nf(x)=" + str(fx) + "\n")
            if rho < 0.25:
                delta = 0.25 * lenp  # reduce the scope of Trust Region
            else:
                delta = min(2 * delta, self.delta_max) if rho > 0.75 and lenp >= delta else delta
                # if reach boundary, expand search area
            x = x + p if rho > self.yita else x
        self.hist = self.hist.append({"x": x, "p": p, "delta": delta, "rho": rho}, ignore_index=True)

    def draw(self):
        x1, x2 = np.meshgrid(np.linspace(-4, 4, 256), np.linspace(-4, 4, 256))
        theta = np.linspace(0, 2*np.pi, 800)
        for i in range(self.hist.shape[0] - 1):
            fig = plt.figure()
            fig.add_subplot(111, aspect='equal')
            plt.contourf(x1, x2, -self.f((x1, x2)), 30)
            plt.scatter(self.hist['x'][i][0], self.hist['x'][i][1], c='red')  # point x at iteration i
            plt.scatter(self.hist['x'][i+1][0], self.hist['x'][i+1][1], c='cyan', marker='v')  # point x at iter i+1
            c1 = self.hist['x'][i][0] + self.hist['delta'][i] * np.cos(theta)
            c2 = self.hist['x'][i][1] + self.hist['delta'][i] * np.sin(theta)
            plt.plot(c1, c2, 'pink')  # draw Trust Region Area.
            # plt.show()
            plt.savefig(str(i) + ".png", dpi=300)


if __name__ == "__main__":
    tr = TrustRegion()
    tr.calc()
    tr.draw()
