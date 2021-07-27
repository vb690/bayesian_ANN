import pymc3 as pm
import theano

from .networks_utils import AbstractLayer


class Dense(AbstractLayer):
    """
    """
    def __init__(self, units, layer_name, prior, activation='sigmoid',
                 weight_init_func='gaussian', bias_init_func='gaussian',
                 **priors_kwargs):
        """
        """
        self.units = units
        self.layer_name = layer_name
        self.weight_init_func = getattr(self, weight_init_func)
        self.bias_init_func = getattr(self, bias_init_func)
        self.activation = getattr(self, activation)
        self.prior = prior
        self.priors_kwargs = priors_kwargs

    def __dense_init(self, shape_in):
        """
        """
        floatX = theano.config.floatX
        if self.units == 1:
            shape = (shape_in,)
        else:
            shape = (shape_in, self.units)

        weights_init = self.weight_init_func(
            shape=shape
        ).astype(floatX)

        biases_init = self.bias_init_func(
            shape=self.units
        )
        return weights_init, biases_init

    def __dense_weights(self, shape_in, weights_init, biases_init):
        """
        """
        if self.units == 1:
            shape = (shape_in,)
        else:
            shape = (shape_in, self.units)

        weights = self.prior(
            f'dense_weights_{self.layer_name}',
            shape=shape,
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
        shape_in = input_tensor.shape.eval()[1]
        weights_init, biases_init = self.__dense_init(
            shape_in,
            shape_out=self.units
        )
        weights, biases = self.__dense_weights(
            shape_in,
            weights_init,
            biases_init
        )
        output_tensor = pm.Deterministic(
                f'dense_{self.layer_name}',
                self.activation_function(
                    pm.math.dot(input_tensor, weights) + biases
                )
        )
        return output_tensor

    class Embedding(AbstractLayer):
        """
        """
        def __init__(self, units, layer_name, prior, activation='sigmoid',
                     embedding_init_func='gaussian', **priors_kwargs):
            """
            """
            self.units = units
            self.layer_name = layer_name
            self.embedding_init_func = getattr(self, embedding_init_func)
            self.activation = getattr(self, activation)
            self.prior = prior
            self.priors_kwargs = priors_kwargs

        def __embedding_init(self, shape_in):
            """
            """
            floatX = theano.config.floatX
            weights_init = self.embedding_init_func(
                shape=(shape_in, self.units)
            ).astype(floatX)
            return weights_init

        def __embedding_weights(self, shape_in, embedding_init):
            """
            """
            weights = self.prior(
                f'embedding_weights_{self.layer_name}',
                shape=(shape_in, self.units),
                testval=embedding_init,
                **self.priors_kwargs
            )
            return weights

        def build(self, input_tensor):
            """
            """
            shape_in = input_tensor.shape.eval()[1]
            embedding_init = self.__embedding_init(
                shape_in=shape_in
            )
            weights = self.__embedding_weights(
                shape_in=shape_in,
                embedding_init=embedding_init
            )

            output_tensor = pm.Deterministic(
                f'embedding_{self.layer_name}',
                self.activation_function(
                    weights
                )[self.input_tensor]
            )
            return output_tensor

    class GRU:
        """
        """
        def __init__(self, units, layer_name, prior,
                     weight_init_func='gaussian', bias_init_func='gaussian',
                     **priors_kwargs):
            self.units = units
            self.layer_name = layer_name
            self.weight_init_func = getattr(self, weight_init_func)
            self.bias_init_func = getattr(self, bias_init_func)
            self.prior = prior
            self.priors_kwargs = priors_kwargs
