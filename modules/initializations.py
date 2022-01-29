import numpy as np


class Initializations:
    """
    """
    def __init__(self,
                 uniform_init_para={"low": -1, "high": 1},
                 gaussian_init_para={"mu": 0, "sigma": 1},
                 laplace_init_para={"mu": 0, "beta": 1}
                 ):
        self.uniform_init_para = uniform_init_para
        self.gaussian_init_para = gaussian_init_para
        self.laplace_init_para = laplace_init_para

    def uniform(self, shape, low=None, high=None):
        """
        """
        if low is None:
            low = self.uniform['low']
        if high is None:
            high = self.uniform['high']

        init = np.random.uniform(
            low=low,
            high=high,
            size=shape
        )
        return init

    def ones(self, shape):
        """
        """
        init = np.ones(
            shape=shape
        )
        return init

    def gaussian(self, shape, mu=0, sigma=1):
        """
        """
        if mu is None:
            mu = self.uniform['mu']
        if sigma is None:
            sigma = self.uniform['sigma']

        init = np.random.normal(
            loc=mu,
            scale=sigma,
            size=shape
        )
        return init

    def gorlot_uniform(self, shape):
        """
        """
        if len(shape) == 1:
            denominator = shape[0]
        else:
            denominator = shape[0] + shape[1]

        limit = np.sqrt(
            6 / denominator
        )
        init = np.random.uniform(
            low=-limit,
            high=limit,
            size=shape
        )
        return init

    def laplace(self, shape, mu=None, beta=None):
        """
        """
        if mu is None:
            mu = self.uniform['mu']
        if beta is None:
            beta = self.uniform['beta']
        init = np.random.laplace(
            loc=mu,
            scale=beta,
            size=shape
        )
        return init
