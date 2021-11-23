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
        """
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
            mu_weights + weights_offset * sigma_weights
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


class RNN(_AbstractLayer):
    """
    """
    def __init__(self, shapes, units, layer_name, prior,
                 return_sequences=False, weight_init_func='gaussian',
                 bias_init_func='gaussian', **priors_kwargs):
        """
        """
        self.units = units
        self.return_sequences = return_sequences

        # ugly shpe specification
        self.shape_batch = shapes[0]
        self.shape_seq = shapes[1]
        self.shape_feat = shapes[2]

        self.layer_name = layer_name
        self.weight_init_func = getattr(self, weight_init_func)
        self.bias_init_func = getattr(self, bias_init_func)
        self.prior = prior
        self.priors_kwargs = priors_kwargs

    def __RNN_init(self):
        """
        """
        floatX = theano.config.floatX

        hs_init = self.weight_init_func(
            shape=(self.shape_batch, self.units)
        ).astype(floatX)
        iw_init = self.weight_init_func(
            shape=(self.shape_feat, self.units)
        ).astype(floatX)
        hw_init = self.weight_init_func(
            shape=(self.units, self.units)
        ).astype(floatX)
        hb_init = self.bias_init_func(
            shape=self.units
        )

        return hs_init, iw_init, hw_init, hb_init

    def __RNN_weights(self, *args):
        """
        """
        hs_init, iw_init, hw_init, hb_init = args[0]

        hs = pm.prior(
            'initial_hidden_state',
            shape=(self.shape_batch, self.units),
            testval=hs_init,
            **self.priors_kwargs
        )
        iw = self.prior(
            f'input_weights_{self.layer_name}',
            shape=(self.shape_feat, self.units),
            testval=iw_init,
            **self.priors_kwargs
        )
        hw = self.prior(
            f'hidden_weights_{self.layer_name}',
            shape=(self.units, self.units),
            testval=hw_init,
            **self.priors_kwargs
        )
        hb = self.prior(
            f'hidden_biases_{self.layer_name}',
            shape=self.units,
            testval=hb_init,
            **self.priors_kwargs
        )

        return hs, iw, hw, hb

    def build(self, input_tensor):
        """
        """
        hs, iw, hw, hb = self.__RNN_weights(
            self.__RNN_init()
        )
        hidden_states = [hs]

        for step in range(self.shape_seq):

            inp = pm.math.dot(input_tensor[:, step, :], iw)
            h = pm.math.dot(hidden_states[step], hw)
            a = hb + inp + h

            hidden_state = pm.Deterministic(
                f'{step}_hidden_state',
                pm.math.tanh(
                    a
                )
            )

            hidden_states.append(hidden_state)

        if self.return_sequences:
            raise NotImplementedError
        else:
            return hidden_states[-1]
