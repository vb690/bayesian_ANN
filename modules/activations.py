import pymc3 as pm

import theano.tensor as T


class Activations:
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

    def log(self, x):
        """Wrapper on pymc3 log function.
        """
        return pm.math.log(x)

    def exp(self, x):
        """Wrapper on pymc3 exp function.
        """
        return pm.exp(x)
