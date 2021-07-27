import numpy as np


class Initializations:
    """
    """
    def __init__(self):
        pass

    def uniform(self, shape, low=-1, high=1):
        """
        """
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

    def laplace(self, shape, mu, beta):
        """
        """
        init = np.random.laplace(
            loc=mu,
            scale=beta,
            size=shape
        )
        return init
