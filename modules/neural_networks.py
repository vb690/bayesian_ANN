import pymc3 as pm
import theano

from modules.networks_utils import AbstractNNet


class BayesianMLP(AbstractNNet):
    """
    """
    def __init__(self, X, y, shape_out, likelyhood_model, prior,
                 layers=(100, 50, 25), activation='tanh',
                 weight_init_func='gaussian', bias_init_func='gaussian',
                 batch_size=32, **priors_kwargs):
        self.layers = layers
        self.activation = activation
        self.shape_out = shape_out
        self.weight_init_func = weight_init_func
        self.bias_init_func = bias_init_func
        self.batch_size = batch_size
        self.prior = prior
        self.__build(
            X=X,
            y=y,
            likelyhood_model=likelyhood_model,
            **priors_kwargs
        )

    def __build(self, X, y, likelyhood_model, **priors_kwargs):
        """
        """
        with pm.Model() as model:

            setattr(
                self,
                'X_data',
                theano.shared(X.astype(X.dtype))
            )
            setattr(
                self,
                'y_data',
                theano.shared(y.astype(y.dtype))
            )
            setattr(
                self,
                'map_tensor_batch',
                {
                    self.X_data: pm.Minibatch(X, self.batch_size),
                    self.y_data: pm.Minibatch(y, self.batch_size)
                }
            )

            likelyhood_model = getattr(
                self,
                likelyhood_model
            )

            for layer_n, units in enumerate(self.layers):

                if layer_n == 0:
                    shape_in = X.shape[1]
                    layer = self.X_data
                else:
                    shape_in = self.layers[layer_n - 1]

                layer = self.dense_layer(
                    layer_name=layer_n,
                    units=units,
                    shape_in=shape_in,
                    previous_layer=layer,
                    weight_init_func=self.weight_init_func,
                    bias_init_func=self.bias_init_func,
                    prior=self.prior,
                    activation=self.activation,
                    **priors_kwargs
                )

            out = likelyhood_model(
                previous_layer=layer,
                observed=self.y_data,
                total_size=y.shape[0],
                **priors_kwargs
            )

        setattr(self, 'model', model)


class BayesianWordEmbedding(AbstractNNet):
    """
    """
    def __init__(self, X, y, shape_out, likelyhood_model, prior,
                 vocabulary_size, embedding_size=50, layers=(100, 50, 25),
                 activation='tanh', weight_init_func='gaussian',
                 bias_init_func='gaussian', batch_size=32, **priors_kwargs):
        self.layers = layers
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.activation = activation
        self.shape_out = shape_out
        self.weight_init_func = weight_init_func
        self.bias_init_func = bias_init_func
        self.prior = prior
        self.batch_size = batch_size
        self.__build(
            X=X,
            y=y,
            likelyhood_model=likelyhood_model,
            **priors_kwargs
        )

    def __build(self, X, y, likelyhood_model, **priors_kwargs):
        """
        """
        with pm.Model() as model:

            setattr(
                self,
                'X_data',
                theano.shared(X.astype(X.dtype))
            )
            setattr(
                self,
                'y_data',
                theano.shared(y.astype(y.dtype))
            )
            setattr(
                self,
                'map_tensor_batch',
                {
                    self.X_data: pm.Minibatch(X, self.batch_size),
                    self.y_data: pm.Minibatch(y, self.batch_size)
                }
            )

            likelyhood_model = getattr(
                self,
                likelyhood_model
            )

            embedding = self.embedding_layer(
                layer_name=0,
                units=self.embedding_size,
                shape_in=self.vocabulary_size,
                previous_layer=self.X_data,
                embedding_init_func=self.weight_init_func,
                prior=self.prior,
                activation=self.activation,
                **priors_kwargs
            )

            embedding = pm.Deterministic(
                'flatten',
                embedding.reshape((-1, X.shape[1] * self.embedding_size))
            )

            for layer_n, units in enumerate(self.layers):

                if layer_n == 0:
                    shape_in = X.shape[1] * self.embedding_size
                    layer = embedding
                else:
                    shape_in = self.layers[layer_n - 1]

                layer = self.dense_layer(
                    layer_name=layer_n,
                    units=units,
                    shape_in=shape_in,
                    previous_layer=layer,
                    weight_init_func=self.weight_init_func,
                    bias_init_func=self.bias_init_func,
                    prior=self.prior,
                    activation=self.activation,
                    **priors_kwargs
                )

            out = likelyhood_model(
                previous_layer=layer,
                observed=self.y_data,
                total_size=y.shape[0]
            )

        setattr(self, 'model', model)
