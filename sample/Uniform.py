import numpy as np

class Uniform:
    """
    Parent class for uniform distributions.
    """

    def __init__(self, n_dims):
        self.n_dims = n_dims

    def grad_log_p(self, x):
        """
        :param x: rows are datapoints
        :return: d/dx log p(x)
        """

        x = np.asarray(x)
        assert (x.ndim == 1 and x.size == self.n_dims) or (x.ndim == 2 and x.shape[1] == self.n_dims), 'wrong size'

        return np.zeros_like(x)


class BoxUniform(Uniform):
    """
    Implements a uniform pdf, constrained in a box.
    """

    def __init__(self, simulator):
        """
        :param lower: array with lower limits
        :param upper: array with upper limits
        """


        Uniform.__init__(self, simulator.min.shape[0])

        self.lower = simulator.min.cpu().detach().numpy()
        self.upper = simulator.max.cpu().detach().numpy()
        self.volume = np.prod(self.upper - self.lower)

    def eval(self, x, ii=None, log=True):
        """
        :param x: evaluate at rows
        :param ii: a list of indices to evaluate marginal, if None then evaluates joint
        :param log: whether to return the log prob
        :return: the prob at x rows
        """

        if type(x).__module__ == 'torch':
            x = np.asarray(x.cpu().detach().numpy())
        else:
            x = np.asarray(x)

        if x.ndim == 1:
            return self.eval(x[np.newaxis, :], ii, log)[0]

        if ii is None:

            in_box = np.logical_and(self.lower <= x, x <= self.upper)
            in_box = np.logical_and.reduce(in_box, axis=1)

            if log:
                prob = -float('inf') * np.ones(in_box.size, dtype=float)
                prob[in_box] = -np.log(self.volume)

            else:
                prob = np.zeros(in_box.size, dtype=float)
                prob[in_box] = 1.0 / self.volume

            return prob

        else:
            assert len(ii) > 0, 'list of indices can''t be empty'
            marginal = BoxUniform(self.lower[ii], self.upper[ii])
            return marginal.eval(x, None, log)

    def gen(self, n_samples=None, rng=np.random):
        """
        :param n_samples: int, number of samples to generate
        :return: numpy array, rows are samples. Only 1 sample (vector) if None
        """

        one_sample = n_samples is None
        u = rng.rand(1 if one_sample else n_samples, self.n_dims)
        x = (self.upper - self.lower) * u + self.lower

        return x[0] if one_sample else x

class Prior(BoxUniform):
    """
    A uniform prior over m1, m2, (+/-)sqrt(s1), (+/-)sqrt(s2), arctanh(r).
    """

    def __init__(self, simulator):

        BoxUniform.__init__(self, simulator)