import abc
from abc import ABC

import pymc3 as pm

import theano.tensor as T

from .initialitations import Initializations
from .activations import Activations


class AbstractLayer(ABC, Initializations, Activations):
    """
    """
    def __init__(self):
        pass

    def __call__(self, input_tensor):
        """
        """
        output_tensor = self.build(
            input_tensor=input_tensor
        )
        return output_tensor

    @abc.abstractmethod
    def build(self):
        pass


class AbstractNNet(__LikelyhoodModels):
    """
    """
    def __init__(self):
        pass

    def fit(self, samples=1000, **kwargs):
        """
        """
        with self.model:

            approx = pm.fit(
                more_replacements=self.map_tensor_batch,
                **kwargs
            )

        trace = approx.sample(samples)
        setattr(self, 'trace', trace)
        return None

    def predict(self, X, y, var_names, samples=1000):
        """
        """
        with self.model:

            self.X_data.set_value(X)
            self.y_data.set_value(y)

            post_pred = pm.sample_posterior_predictive(
                self.trace,
                samples=samples,
                var_names=var_names
            )
        return post_pred

    def show_graph(self):
        """
        """
        return pm.model_to_graphviz(self.model)
