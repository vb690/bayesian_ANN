import numpy as np

import theano
import pymc3 as pm

from modules.networks_utils import AbstractNNet


class BayesianMLP(AbstractNNet):
    """
    """
    def __init__(self, X, y, shape_out, likelyhood_model,
                 layers=(100, 50, 25), activation='tanh', **kwargs):
        self.layers = layers
        self.activation = activation
        self.shape_out = shape_out
        self.__build_model(
            X=X,
            y=y,
            likelyhood_model=likelyhood_model,
            **kwargs
        )

    def __weights_init(self, X, y):
        """
        """
        floatX = theano.config.floatX
        weights_inits = {}

        for layer, units in enumerate(self.layers):

            if layer == 0:
                weights_inits[layer] = np.random.randn(
                    X.shape[1],
                    units
                ).astype(floatX)
            else:
                weights_inits[layer] = np.random.randn(
                    self.layers[layer-1],
                    units
                ).astype(floatX)

        if self.shape_out == 1:
            weights_inits['out'] = np.random.randn(
                self.layers[-1]
            ).astype(floatX)
        else:
            weights_inits['out'] = np.random.randn(
                self.layers[-1],
                self.shape_out
            ).astype(floatX)

        return weights_inits

    def __biases_init(self, y):
        """
        """
        floatX = theano.config.floatX
        biases_inits = {}

        for layer, units in enumerate(self.layers):

            biases_inits[layer] = np.random.randn(
                units
            ).astype(floatX)

        biases_inits['out'] = np.random.randn(
            self.shape_out
        ).astype(floatX)

        return biases_inits

    def __create_weights(self, X, y):
        """
        """
        weights = {}
        weights_inits = self.__weights_init(
            X=X,
            y=y
        )

        for layer, units in enumerate(self.layers):

            if layer == 0:
                shape = (X.shape[1], units)
            else:
                shape = (self.layers[layer-1], units)

            weights[layer] = pm.Normal(
                f'weight_{layer}',
                sigma=1,
                mu=0,
                shape=shape,
                testval=weights_inits[layer]
            )

        if self.shape_out == 1:
            shape = (self.layers[-1],)
        else:
            shape = (self.layers[-1], self.shape_out)
        weights['out'] = pm.Normal(
            f'weight_out',
            sigma=1,
            mu=0,
            shape=shape,
            testval=weights_inits['out']
        )

        return weights

    def __create_biases(self, y):
        """
        """
        biases = {}
        biases_inits = self.__biases_init(y=y)

        for layer, units in enumerate(self.layers):

            biases[layer] = pm.Normal(
                f'bias_{layer}',
                sigma=1,
                mu=0,
                shape=units,
                testval=biases_inits[layer]
            )

        biases['out'] = pm.Normal(
            f'bias_out',
            sigma=1,
            mu=0,
            shape=self.shape_out,
            testval=biases_inits['out']
        )

        return biases

    def __build_model(self, X, y, likelyhood_model, **kwargs):
        """
        """
        activation_function = getattr(self, self.activation)
        with pm.Model() as model:

            X_data = pm.Data('X_data', X)
            y_data = pm.Data('y_data', y)

            weights = self.__create_weights(X=X, y=y)
            biases = self.__create_biases(y=y)

            layers = {}
            for layer in range(len(self.layers)):

                if layer == 0:
                    layers[layer] = pm.Deterministic(
                            f'layer_{layer}',
                            activation_function(
                                pm.math.dot(
                                    X_data, weights[layer]
                                ) + biases[layer]
                            )
                    )
                else:
                    layers[layer] = pm.Deterministic(
                            f'layer_{layer}',
                            activation_function(
                                pm.math.dot(
                                    layers[layer-1], weights[layer]
                                ) + biases[layer]
                            )
                    )

            out = likelyhood_model(
                previous_layer=layers[len(self.layers) - 1],
                weights=weights['out'],
                biases=biases['out'],
                observed=y_data,
                total_size=y.shape[0],
                **kwargs
            )

        setattr(self, 'model', model)
