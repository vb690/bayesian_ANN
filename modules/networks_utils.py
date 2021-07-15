import numpy as np

import pymc3 as pm

import theano.tensor as T


class __Activations:
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


class AbstractNNet(__Activations):
    """
    """
    def __init__(self):
        pass

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


def prob_log_loss(y, p):
    """
    """
    y = np.array(
        [y for i in range(p.shape[0])]
    )
    loss = (y*np.log(p) + (1 - y) * np.log(1 - p))
    return -loss
