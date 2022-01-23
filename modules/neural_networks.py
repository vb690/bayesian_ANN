import numpy as np

import pymc3 as pm
import theano

from modules.layers import Dense, Embedding
from .likelyhood_models import LikelyhoodModels


class __AbstractNNet(LikelyhoodModels):
    """
    """
    def __init__(self):
        pass

    def init_graph(self, X, y, likelyhood_model, advi_approx):
        """
        """
        if advi_approx:
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
            total_size = y.shape[0]
        else:
            setattr(
                self,
                'X_data',
                pm.Data(
                    'X_data',
                    X
                )
            )
            setattr(
                self,
                'y_data',
                pm.Data(
                    'y_data',
                    y
                )
            )
            total_size = None

        if isinstance(likelyhood_model, str):
            likelyhood_model = getattr(
                self,
                likelyhood_model
            )

        return total_size, likelyhood_model

    def fit(self, samples=1000, **kwargs):
        """
        """
        with self.model:

            if self.advi_approx:
                approx = pm.fit(
                    more_replacements=self.map_tensor_batch,
                    **kwargs
                )

                trace = approx.sample(samples)
            else:
                trace = pm.sample(**kwargs)

        setattr(self, 'trace', trace)
        return None

    def get_traces(self):
        """
        """
        return self.trace

    def predict(self, X, y, var_names=None, samples=1000):
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

    def debug(self):
        """
        """
        print(self.model.check_test_point())
        return self.show_graph()


class BayesianMLP(__AbstractNNet):
    """
    """
    def __init__(self, X, y, shape_out, likelyhood_model, prior,
                 layers=(100, 50, 25), activation='tanh', advi_approx=False,
                 weight_init_func='gaussian', bias_init_func='gaussian',
                 batch_size=32, **priors_kwargs):
        self.layers = layers
        self.activation = activation
        self.advi_approx = advi_approx
        self.shape_out = shape_out
        self.weight_init_func = weight_init_func
        self.bias_init_func = bias_init_func
        self.batch_size = batch_size
        self.prior = prior
        self.priors_kwargs = priors_kwargs
        self.__build_graph(
            X=X,
            y=y,
            likelyhood_model=likelyhood_model,
            **priors_kwargs
        )

    def __build_graph(self, X, y, likelyhood_model, **priors_kwargs):
        """
        """
        with pm.Model() as model:

            total_size, likelyhood_model = self.init_graph(
                X=X,
                y=y,
                likelyhood_model=likelyhood_model,
                advi_approx=self.advi_approx
            )

            dense = self.X_data
            shape_in = X.shape[1]
            for layer_n, units in enumerate(self.layers):

                dense = Dense(
                    shape_in=shape_in,
                    units=units,
                    layer_name=layer_n,
                    prior=self.prior,
                    activation=self.activation,
                    **self.priors_kwargs
                )(dense)
                shape_in = units

            out = likelyhood_model(
                shape_in=self.layers[-1],
                input_tensor=dense,
                out_shape=self.shape_out,
                observed=self.y_data,
                total_size=total_size,
                prior=self.prior,
                **self.priors_kwargs
            )

        setattr(self, 'model', model)


class BayesianAutoencoder(__AbstractNNet):
    """
    """
    def __init__(self, X, likelyhood_model, prior,
                 layers=(100, 50), latent_size=25, activation='tanh',
                 weight_init_func='gaussian', bias_init_func='gussian',
                 batch_size=32, denoising=True, noise_sigma=0.1,
                 **priors_kwargs):
        self.layers = layers
        self.latent_size = latent_size
        self.denoising = denoising
        if denoising:
            self.noise_sigma = noise_sigma
        self.activation = activation
        self.shape_out = X.shape[1]
        self.weight_init_func = weight_init_func
        self.bias_init_func = bias_init_func
        self.batch_size = batch_size
        self.prior = prior
        self.priors_kwargs = priors_kwargs
        self.__build_graph(
            X=X,
            likelyhood_model=likelyhood_model,
            **priors_kwargs
        )

    def __build_graph(self, X, likelyhood_model, **priors_kwargs):
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
                theano.shared(X.astype(X.dtype))
            )
            if self.denoising:
                y = X + np.random.normal(
                    0,
                    self.noise_sigma,
                    size=X.shape
                )
            else:
                y = X
            setattr(
                self,
                'map_tensor_batch',
                {
                    self.X_data: pm.Minibatch(X, self.batch_size),
                    self.y_data: pm.Minibatch(y, self.batch_size)
                }
            )

            if isinstance(likelyhood_model, str):
                likelyhood_model = getattr(
                    self,
                    likelyhood_model
                )

            dense = self.X_data
            shape_in = X.shape[1]
            for layer_n, units in enumerate(self.layers):

                dense = Dense(
                    shape_in=shape_in,
                    units=units,
                    layer_name=f'encoder_{layer_n}',
                    prior=self.prior,
                    activation=self.activation,
                    **self.priors_kwargs
                )(dense)
                shape_in = units

            dense = Dense(
                shape_in=shape_in,
                units=self.latent_size,
                layer_name='latent',
                prior=self.prior,
                activation=self.activation,
                **self.priors_kwargs
            )(dense)
            shape_in = self.latent_size

            for layer_n, units in enumerate(self.layers[::-1]):

                dense = Dense(
                    shape_in=shape_in,
                    units=units,
                    layer_name=f'decoder_{layer_n}',
                    prior=self.prior,
                    activation=self.activation,
                    **self.priors_kwargs
                )(dense)
                shape_in = units

            out = likelyhood_model(
                shape_in=units,
                input_tensor=dense,
                out_shape=self.shape_out,
                observed=self.y_data,
                total_size=y.shape[0],
                prior=self.prior,
                **self.priors_kwargs
            )

        setattr(self, 'model', model)


class BayesianWordEmbedding(__AbstractNNet):
    """
    """
    def __init__(self, X, y, shape_out, likelyhood_model, prior,
                 vocabulary_size, mu=0, sigma=10, beta=5,
                 embedding_size=50, layers=(100, 50, 25),
                 activation='tanh', weight_init_func='gaussian',
                 bias_init_func='gaussian', batch_size=32, priors_dict={
                    'mu': 0,
                    'sigma': 10}
                 ):
        self.layers = layers
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.activation = activation
        self.shape_out = shape_out
        self.weight_init_func = weight_init_func
        self.bias_init_func = bias_init_func
        self.batch_size = batch_size
        self.mu = mu
        self.prior = prior
        self.sigma = sigma
        self.beta = beta
        self.priors_dict = priors_dict
        self.__build_graph(
            X=X,
            y=y,
            likelyhood_model=likelyhood_model,
            **self.priors_dict
        )

    def __build_graph(self, X, y, likelyhood_model, **priors_dict):
        """
        """
        with pm.Model() as model:

            total_size, likelyhood_model = self.init_graph(
                X=X,
                y=y,
                likelyhood_model=likelyhood_model,
                advi_approx=self.advi_approx
            )

            embedding = Embedding(
                vocabulary_size=self.vocabulary_size,
                units=self.embedding_size,
                layer_name=0,
                mu=self.mu,
                sigma=self.sigma,
                beta=self.beta
            )(self.X_data)

            embedding = pm.Deterministic(
                'flatten',
                embedding.reshape((-1, X.shape[1] * self.embedding_size))
            )

            for layer_n, units in enumerate(self.layers):

                if layer_n == 0:
                    shape_in = X.shape[1] * self.embedding_size
                    dense = embedding
                else:
                    shape_in = self.layers[layer_n - 1]

                dense = Dense(
                    shape_in=shape_in,
                    units=units,
                    layer_name=layer_n,
                    prior=self.prior,
                    activation=self.activation,
                    **self.priors_dict
                )(dense)

            out = likelyhood_model(
                shape_in=self.layers[-1],
                input_tensor=dense,
                out_shape=self.shape_out,
                observed=self.y_data,
                total_size=y.shape[0],
                prior=self.prior,
                **self.priors_dict
            )

        setattr(self, 'model', model)
