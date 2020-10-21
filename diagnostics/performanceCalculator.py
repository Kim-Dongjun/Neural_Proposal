import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def gaussian_kde(xs, ws=None, std=None):
    """
    Returns a mixture of gaussians representing a kernel density estimate.
    :param xs: rows are datapoints
    :param ws: weights, optional
    :param std: the std of the kernel, if None then a default is used
    :return: a MoG object
    """

    xs = np.array(xs)
    assert xs.ndim == 2, 'wrong shape'

    n_data, n_dims = xs.shape
    ws = np.full(n_data, 1.0 / n_data) if ws is None else np.asarray(ws)
    var = n_data ** (-2.0 / (n_dims + 4)) if std is None else std ** 2

    return MoG(a=ws, ms=xs, Ss=[var * np.eye(n_dims) for _ in range(n_data)])

class MoG:
    """
    Implements a mixture of gaussians.
    """

    def __init__(self, a, ms=None, Ps=None, Us=None, Ss=None, xs=None):
        """
        Creates a mog with a valid combination of parameters or an already given list of gaussian variables.
        :param a: mixing coefficients
        :param ms: means
        :param Ps: precisions
        :param Us: precision factors such that U'U = P
        :param Ss: covariances
        :param xs: list of gaussian variables
        """

        if ms is not None:

            if Ps is not None:
                self.xs = [Gaussian(m=m, P=P) for m, P in zip(ms, Ps)]

            elif Us is not None:
                self.xs = [Gaussian(m=m, U=U) for m, U in zip(ms, Us)]

            elif Ss is not None:
                self.xs = [Gaussian(m=m, S=S) for m, S in zip(ms, Ss)]

            else:
                raise ValueError('Precision information missing.')

        elif xs is not None:
            self.xs = xs

        else:
            raise ValueError('Mean information missing.')

        self.a = np.asarray(a)
        self.n_dims = self.xs[0].n_dims
        self.n_components = len(self.xs)

    def gen(self, n_samples=None, return_comps=False, rng=np.random):
        """
        Generates independent samples from mog.
        """

        if n_samples is None:

            i = discrete_sample(self.a, rng=rng)
            sample = self.xs[i].gen(rng=rng)

            return (sample, i) if return_comps else sample

        else:

            samples = np.empty([n_samples, self.n_dims])
            ii = discrete_sample(self.a, n_samples, rng)
            for i, x in enumerate(self.xs):
                idx = ii == i
                N = np.sum(idx.astype(int))
                samples[idx] = x.gen(N, rng=rng)

            return (samples, ii) if return_comps else samples

    def eval(self, x, ii=None, log=True):
        """
        Evaluates the mog pdf.
        :param x: rows are inputs to evaluate at
        :param ii: a list of indices specifying which marginal to evaluate. if None, the joint pdf is evaluated
        :param log: if True, the log pdf is evaluated
        :return: pdf or log pdf
        """

        x = np.asarray(x)

        if x.ndim == 1:
            return self.eval(x[np.newaxis, :], ii, log)[0]

        ps = np.array([c.eval(x, ii, log) for c in self.xs]).T
        res = scipy.special.logsumexp(ps + np.log(self.a), axis=1) if log else np.dot(ps, self.a)

        return res

    def grad_log_p(self, x):
        """
        Evaluates the gradient of the log mog pdf.
        :param x: rows are inputs to evaluate at
        :return: d/dx log p(x)
        """

        x = np.asarray(x)

        if x.ndim == 1:
            return self.grad_log_p(x[np.newaxis, :])[0]

        ps = np.array([c.eval(x, log=True) for c in self.xs])
        ws = np.exp(logsoftmax(ps.T + np.log(self.a))).T
        ds = np.array([c.grad_log_p(x) for c in self.xs])

        res = np.sum(ws[:, :, np.newaxis] * ds, axis=0)

        return res

    def __mul__(self, other):
        """
        Multiply with a single gaussian.
        """

        assert isinstance(other, Gaussian)

        ys = [x * other for x in self.xs]

        lcs = np.empty_like(self.a)

        for i, (x, y) in enumerate(zip(self.xs, ys)):

            lcs[i] = x.logdetP + other.logdetP - y.logdetP
            lcs[i] -= np.dot(x.m, np.dot(x.P, x.m)) + np.dot(other.m, np.dot(other.P, other.m)) - np.dot(y.m, np.dot(y.P, y.m))
            lcs[i] *= 0.5

        la = np.log(self.a) + lcs
        la -= scipy.special.logsumexp(la)
        a = np.exp(la)

        return MoG(a=a, xs=ys)

    def __imul__(self, other):
        """
        Incrementally multiply with a single gaussian.
        """

        assert isinstance(other, Gaussian)

        res = self * other

        self.a = res.a
        self.xs = res.xs

        return res

    def __div__(self, other):
        """
        Divide by a single gaussian.
        """

        assert isinstance(other, Gaussian)

        ys = [x / other for x in self.xs]

        lcs = np.empty_like(self.a)

        for i, (x, y) in enumerate(zip(self.xs, ys)):

            lcs[i] = x.logdetP - other.logdetP - y.logdetP
            lcs[i] -= np.dot(x.m, np.dot(x.P, x.m)) - np.dot(other.m, np.dot(other.P, other.m)) - np.dot(y.m, np.dot(y.P, y.m))
            lcs[i] *= 0.5

        la = np.log(self.a) + lcs
        la -= scipy.special.logsumexp(la)
        a = np.exp(la)

        return MoG(a=a, xs=ys)

    def __idiv__(self, other):
        """
        Incrementally divide by a single gaussian.
        """

        assert isinstance(other, Gaussian)

        res = self / other

        self.a = res.a
        self.xs = res.xs

        return res

    def calc_mean_and_cov(self):
        """
        Calculate the mean vector and the covariance matrix of the mog.
        """

        ms = [x.m for x in self.xs]
        m = np.dot(self.a, np.array(ms))

        msqs = [x.S + np.outer(mi, mi) for x, mi in zip(self.xs, ms)]
        S = np.sum(np.array([a * msq for a, msq in zip(self.a, msqs)]), axis=0) - np.outer(m, m)

        return m, S

    def project_to_gaussian(self):
        """
        Returns a gaussian with the same mean and precision as the mog.
        """

        m, S = self.calc_mean_and_cov()
        return Gaussian(m=m, S=S)

    def prune_negligible_components(self, threshold):
        """
        Removes all the components whose mixing coefficient is less than a threshold.
        """

        ii = np.nonzero((self.a < threshold).astype(int))[0]
        total_del_a = np.sum(self.a[ii])
        del_count = ii.size

        self.n_components -= del_count
        self.a = np.delete(self.a, ii)
        self.a += total_del_a / self.n_components
        self.xs = [x for i, x in enumerate(self.xs) if i not in ii]

    def kl(self, other, n_samples=10000, rng=np.random):
        """
        Estimates the kl from this to another pdf, i.e. KL(this | other), using monte carlo.
        """

        x = self.gen(n_samples, rng=rng)
        lp = self.eval(x, log=True)
        lq = other.eval(x, log=True)
        t = lp - lq

        res = np.mean(t)
        err = np.std(t, ddof=1) / np.sqrt(n_samples)

        return res, err

class Gaussian:
    """
    Implements a gaussian pdf. Focus is on efficient multiplication, division and sampling.
    """

    def __init__(self, m=None, P=None, U=None, S=None, Pm=None):
        """
        Initialize a gaussian pdf given a valid combination of its parameters. Valid combinations are:
        m-P, m-U, m-S, Pm-P, Pm-U, Pm-S
        :param m: mean
        :param P: precision
        :param U: upper triangular precision factor such that U'U = P
        :param S: covariance
        :param Pm: precision times mean such that P*m = Pm
        """

        try:
            if m is not None:
                m = np.asarray(m)
                self.m = m
                self.n_dims = m.size

                if P is not None:
                    P = np.asarray(P)
                    L = np.linalg.cholesky(P)
                    self.P = P
                    self.C = np.linalg.inv(L)
                    self.S = np.dot(self.C.T, self.C)
                    self.Pm = np.dot(P, m)
                    self.logdetP = 2.0 * np.sum(np.log(np.diagonal(L)))

                elif U is not None:
                    U = np.asarray(U)
                    self.P = np.dot(U.T, U)
                    self.C = np.linalg.inv(U.T)
                    self.S = np.dot(self.C.T, self.C)
                    self.Pm = np.dot(self.P, m)
                    self.logdetP = 2.0 * np.sum(np.log(np.diagonal(U)))

                elif S is not None:
                    S = np.asarray(S)
                    self.P = np.linalg.inv(S)
                    self.C = np.linalg.cholesky(S).T
                    self.S = S
                    self.Pm = np.dot(self.P, m)
                    self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))

                else:
                    raise ValueError('Precision information missing.')

            elif Pm is not None:
                Pm = np.asarray(Pm)
                self.Pm = Pm
                self.n_dims = Pm.size

                if P is not None:
                    P = np.asarray(P)
                    L = np.linalg.cholesky(P)
                    self.P = P
                    self.C = np.linalg.inv(L)
                    self.S = np.dot(self.C.T, self.C)
                    self.m = np.linalg.solve(P, Pm)
                    self.logdetP = 2.0 * np.sum(np.log(np.diagonal(L)))

                elif U is not None:
                    U = np.asarray(U)
                    self.P = np.dot(U.T, U)
                    self.C = np.linalg.inv(U.T)
                    self.S = np.dot(self.C.T, self.C)
                    self.m = np.linalg.solve(self.P, Pm)
                    self.logdetP = 2.0 * np.sum(np.log(np.diagonal(U)))

                elif S is not None:
                    S = np.asarray(S)
                    self.P = np.linalg.inv(S)
                    self.C = np.linalg.cholesky(S).T
                    self.S = S
                    self.m = np.dot(S, Pm)
                    self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))

                else:
                    raise ValueError('Precision information missing.')

            else:
                raise ValueError('Mean information missing.')

        except np.linalg.LinAlgError:
            raise ImproperCovarianceError()

    def gen(self, n_samples=None, rng=np.random):
        """
        Returns independent samples from the gaussian.
        """

        one_sample = n_samples is None

        z = rng.randn(1 if one_sample else n_samples, self.n_dims)
        samples = np.dot(z, self.C) + self.m

        return samples[0] if one_sample else samples

    def eval(self, x, ii=None, log=True):
        """
        Evaluates the gaussian pdf.
        :param x: rows are inputs to evaluate at
        :param ii: a list of indices specifying which marginal to evaluate. if None, the joint pdf is evaluated
        :param log: if True, the log pdf is evaluated
        :return: pdf or log pdf
        """

        x = np.asarray(x)

        if x.ndim == 1:
            return self.eval(x[np.newaxis, :], ii, log)[0]

        if ii is None:
            xm = x - self.m
            lp = -np.sum(np.dot(xm, self.P) * xm, axis=1)
            lp += self.logdetP - self.n_dims * np.log(2.0 * np.pi)
            lp *= 0.5

        else:
            m = self.m[ii]
            S = self.S[ii][:, ii]
            lp = scipy.stats.multivariate_normal.logpdf(x, m, S)
            lp = np.array([lp]) if x.shape[0] == 1 else lp

        return lp if log else np.exp(lp)

    def grad_log_p(self, x):
        """
        Evaluates the gradient of the log pdf.
        :param x: rows are inputs to evaluate at
        :return: d/dx log p(x)
        """

        x = np.asarray(x)
        return -np.dot(x - self.m, self.P)

    def __mul__(self, other):
        """
        Multiply with another gaussian.
        """

        assert isinstance(other, Gaussian)

        P = self.P + other.P
        Pm = self.Pm + other.Pm

        return Gaussian(P=P, Pm=Pm)

    def __imul__(self, other):
        """
        Incrementally multiply with another gaussian.
        """

        assert isinstance(other, Gaussian)

        res = self * other

        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP

        return res

    def __div__(self, other):
        """
        Divide by another gaussian. Note that the resulting gaussian might be improper.
        """

        assert isinstance(other, Gaussian)

        P = self.P - other.P
        Pm = self.Pm - other.Pm

        return Gaussian(P=P, Pm=Pm)

    def __idiv__(self, other):
        """
        Incrementally divide by another gaussian. Note that the resulting gaussian might be improper.
        """

        assert isinstance(other, Gaussian)

        res = self / other

        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP

        return res

    def __pow__(self, power, modulo=None):
        """
        Raise gaussian to a power and get another gaussian.
        """

        P = power * self.P
        Pm = power * self.Pm

        return Gaussian(P=P, Pm=Pm)

    def __ipow__(self, power):
        """
        Incrementally raise gaussian to a power.
        """

        res = self ** power

        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP

        return res

    def kl(self, other):
        """
        Calculates the kl divergence from this to another gaussian, i.e. KL(this | other).
        """

        assert isinstance(other, Gaussian)
        assert self.n_dims == other.n_dims

        t1 = np.sum(other.P * self.S)

        m = other.m - self.m
        t2 = np.dot(m, np.dot(other.P, m))

        t3 = self.logdetP - other.logdetP

        t = 0.5 * (t1 + t2 + t3 - self.n_dims)

        return t

class ImproperCovarianceError(Exception):
    """
    Exception to be thrown when a Gaussian is created with a covariance matrix that isn't strictly positive definite.
    """

    def __str__(self):
        return 'Covariance matrix is not strictly positive definite'

def discrete_sample(p, n_samples=None, rng=np.random):
    """
    Samples from a discrete distribution.
    :param p: a distribution with N elements
    :param n_samples: number of samples, only 1 if None
    :return: vector of samples
    """

    # check distribution
    # assert isdistribution(p), 'Probabilities must be non-negative and sum to one.'

    one_sample = n_samples is None

    # cumulative distribution
    c = np.cumsum(p[:-1])[np.newaxis, :]

    # get the samples
    r = rng.rand(1 if one_sample else n_samples, 1)
    samples = np.sum((r > c).astype(int), axis=1)

    return samples[0] if one_sample else samples

def logsoftmax(x):
    """
    Calculates the log softmax of x, or of the rows of x.
    :param x: vector or matrix
    :return: log softmax
    """

    x = np.asarray(x)

    if x.ndim == 1:
        return x - scipy.special.logsumexp(x)

    elif x.ndim == 2:
        return x - scipy.special.logsumexp(x, axis=1)[:, np.newaxis]

    else:
        raise ValueError('input must be either vector or matrix')