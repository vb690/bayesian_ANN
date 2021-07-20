import numpy as np

import pymc3 as pm

import theano
import theano.tensor as T


class __Activations:
    """
    """
    def __init__(self):
        pass

    def linear(self, x):
        """Linear activation function.
        """
        return x

    def sigmoid(self, x):
        """Wrapper on pymc3 sigmoid function.
        """
        return pm.math.sigmoid(x)

    def tanh(self, x):
        """Wrapper on pymc3 tanh function.
        """
        return pm.math.tanh(x)

    def relu(self, x):
        """Wrapper on Theano relu function.
        """
        return T.nnet.relu(x)

    def softmax(self, x):
        """Wrapper on Theano softmax function.
        """
        return T.nnet.softmax(x)


class __Initializations:
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


class AbstractNNet(__Activations, __Initializations):
    """
    """
    def __init__(self):
        pass

    def __dense_init(self, shape_in, shape_out, weight_init_func,
                     bias_init_func):
        """
        """
        floatX = theano.config.floatX
        if shape_out == 1:
            shape = (shape_in,)
        else:
            shape = (shape_in, shape_out)

        weights_init = weight_init_func(
            shape=shape
        ).astype(floatX)

        biases_init = bias_init_func(
            shape=shape_out
        )
        return weights_init, biases_init

    def __embedding_init(self, shape_in, shape_out, embedding_init_func):
        """
        """
        floatX = theano.config.floatX
        weights_init = embedding_init_func(
            shape=(shape_in, shape_out)
        ).astype(floatX)
        return weights_init

    def __dense_weights(self, shape_in, shape_out, layer_name, prior,
                        weights_init, biases_init, **priors_kwargs):
        """
        """
        if shape_out == 1:
            shape = (shape_in,)
        else:
            shape = (shape_in, shape_out)

        weights = prior(
            f'dense_weights_{layer_name}',
            shape=shape,
            testval=weights_init,
            **priors_kwargs
        )
        biases = prior(
            f'dense_biases_{layer_name}',
            shape=shape_out,
            testval=biases_init,
            **priors_kwargs
        )
        return weights, biases

    def __embedding_weights(self, shape_in, shape_out, layer_name, prior,
                            embedding_init, **priors_kwargs):
        """
        """
        weights = prior(
            f'embedding_weights_{layer_name}',
            shape=(shape_in, shape_out),
            testval=embedding_init,
            **priors_kwargs
        )
        return weights

    def dense_layer(self, layer_name, units, shape_in, previous_layer,
                    weight_init_func, bias_init_func, prior, activation,
                    **priors_kwargs):
        """
        """
        activation_function = getattr(self, activation)
        weight_init_func = getattr(self, weight_init_func)
        bias_init_func = getattr(self, bias_init_func)

        weights_init, biases_init = self.__dense_init(
            shape_in=shape_in,
            shape_out=units,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func
        )
        weights, biases = self.__dense_weights(
            shape_in=shape_in,
            shape_out=units,
            layer_name=layer_name,
            prior=prior,
            weights_init=weights_init,
            biases_init=biases_init,
            **priors_kwargs
        )

        layer = pm.Deterministic(
                f'dense_{layer_name}',
                activation_function(
                    pm.math.dot(previous_layer, weights) + biases
                )
        )
        return layer

    def embedding_layer(self, layer_name, units, shape_in, previous_layer,
                        embedding_init_func, prior, activation,
                        **priors_kwargs):
        """
        """
        activation_function = getattr(self, activation)
        embedding_init_func = getattr(self, embedding_init_func)

        embedding_init = self.__embedding_init(
            shape_in=shape_in,
            shape_out=units,
            embedding_init_func=embedding_init_func
        )
        weights = self.__embedding_weights(
            shape_in=shape_in,
            shape_out=units,
            layer_name=layer_name,
            prior=prior,
            embedding_init=embedding_init,
            **priors_kwargs
        )

        layer = pm.Deterministic(
            f'embedding_{layer_name}',
            activation_function(
                weights
            )
        )
        return layer

    def fit(self, samples=1000, **kwargs):
        """
        """
        with self.model:

            approx = pm.fit(**kwargs)

        trace = approx.sample(samples)
        setattr(self, 'trace', trace)
        return None

    def predict(self, X, y, var_names, samples=1000):
        """
        """
        with self.model:

            pm.set_data(
                {
                    'X_data': X,
                    'y_data': y
                }
            )
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
