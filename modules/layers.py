import abc
from abc import ABC

import pymc3 as pm
import theano

from .initializations import Initializations
from .activations import Activations


class _AbstractLayer(ABC, Initializations, Activations):
    """
    """
    def __init__(self):
        super(Initializations, self).__init__()

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


class Dense(_AbstractLayer):
    """
    """
    def __init__(self, shape_in, units, layer_name, prior,
                 activation='sigmoid', weight_init_func='gaussian',
                 bias_init_func='gaussian', **priors_kwargs):
        """
        """
        self.shape = (shape_in, units) if units > 1 else (shape_in, )
        self.units = units
        self.layer_name = layer_name
        self.weight_init_func = getattr(self, weight_init_func)
        self.bias_init_func = getattr(self, bias_init_func)
        self.activation = getattr(self, activation)
        self.prior = prior
        self.priors_kwargs = priors_kwargs

    def __dense_init(self):
        """
        """
        floatX = theano.config.floatX
        weights_init = self.weight_init_func(
            shape=self.shape
        ).astype(floatX)
        biases_init = self.bias_init_func(
            shape=self.units
        )
        return weights_init, biases_init

    def __dense_weights(self, *args):
        """
        """
        weights_init, biases_init = args[0]

        weights = self.prior(
            f'dense_weights_{self.layer_name}',
            shape=self.shape,
            testval=weights_init,
            **self.priors_kwargs
        )
        biases = self.prior(
            f'dense_biases_{self.layer_name}',
            shape=self.units,
            testval=biases_init,
            **self.priors_kwargs
        )
        return weights, biases

    def build(self, input_tensor):
        """
        """
        weights, biases = self.__dense_weights(
            self.__dense_init()
        )
        output_tensor = pm.Deterministic(
                f'dense_{self.layer_name}',
                self.activation(
                    pm.math.dot(input_tensor, weights) + biases
                )
        )
        return output_tensor


class Embedding(_AbstractLayer):
    """
    """
    def __init__(self, vocabulary_size, units, layer_name, mu, sigma, beta,
                 activation='sigmoid', embedding_init_func='gaussian'):
        """
        """
        # we reserve 0 as a special number for copturing the variations
        # of all undefined categories. This should regress to the mean of the
        # hierarchical distribution.
        self.vocabulary_size = vocabulary_size + 1
        self.units = units
        self.layer_name = layer_name
        self.embedding_init_func = getattr(self, embedding_init_func)
        self.activation = getattr(self, activation)
        self.mu = mu
        self.sigma = sigma
        self.beta = beta

    def __embedding_init(self):
        """
        """
        floatX = theano.config.floatX
        weights_init = self.embedding_init_func(
            shape=(self.vocabulary_size, self.units)
        ).astype(floatX)
        return weights_init

    def __embedding_weights(self):
        """We construnct an embedding layer as a lookup table
        ----------------------
        | Cat Index |   W    |
        ----------------------
        |     0     | [.....]|
        |     1     | [.....]|
        |     2     | [.....]|
        |     3     | [.....]|

        each W comes from a hierarchical structure.The template we followed
        is from this blogpost
        https://twiecki.io/blog/2018/08/13/hierarchical_bayesian_neural_network
        """
        floatX = theano.config.floatX

        weights_offset = pm.Normal(
            f'embedding_weights_offset_{self.layer_name}',
            mu=0,
            sd=1,
            testval=self.embedding_init_func(
                shape=(self.vocabulary_size, self.units)
            ).astype(floatX),
            shape=(self.vocabulary_size, self.units)
        )
        mu_weights = pm.Normal(
            f'embedding_weights_mu_{self.layer_name}',
            mu=self.mu,
            sd=self.sigma,
            testval=self.embedding_init_func(
                shape=(self.units)
            ).astype(floatX),
            shape=(self.units)
        )
        sigma_weights = pm.HalfCauchy(
            f'embedding_weights_sigma_{self.layer_name}',
            self.beta
        )
        weights = pm.Deterministic(
            f'embedding_weights_{self.layer_name}',
            weights_offset * sigma_weights + mu_weights
        )
        return weights

    def build(self, input_tensor):
        """
        """
        weights = self.__embedding_weights()
        output_tensor = pm.Deterministic(
            f'embedding_{self.layer_name}',
            self.activation(
                weights
            )[input_tensor]
        )
        return output_tensor
