import pymc3 as pm
import theano.tensor as T

from .layers import Dense


class LikelyhoodModels:

    def __init__(self):
        """
        """
        pass

    def gaussian_lk(self, shape_in, input_tensor, out_shape, observed,
                    total_size, prior, beta=.1, **priors_kwargs):
        """
        """
        with pm.Model() as lk_model:

            mu = Dense(
                shape_in=shape_in,
                units=out_shape,
                layer_name='mu',
                prior=prior,
                activation='linear',
                **priors_kwargs
            )(input_tensor)

            sd = pm.HalfCauchy(
                name='sigma',
                beta=beta
            )

            out = pm.Normal(
                'y',
                mu=mu,
                sd=sd,
                observed=observed,
                total_size=total_size,
            )

        return lk_model

    def student_lk(self, shape_in, input_tensor, out_shape, observed,
                   total_size, prior, beta_cauchy=.1, nu=3, **priors_kwargs):
        """
        """
        with pm.Model() as lk_model:

            mu = Dense(
                shape_in=shape_in,
                units=out_shape,
                layer_name='mu',
                prior=prior,
                activation='linear',
                **priors_kwargs
            )(input_tensor)
            sd = pm.HalfCauchy(
                name='sigma',
                beta=beta_cauchy
            )

            out = pm.StudentT(
                'y',
                mu=mu,
                sd=sd,
                nu=nu,
                observed=observed,
                total_size=total_size,
            )

        return lk_model

    def categorical_lk(self, shape_in, input_tensor, out_shape, observed,
                       total_size, prior, **priors_kwargs):
        """
        """
        with pm.Model() as lk_model:

            theta = Dense(
                shape_in=shape_in,
                units=out_shape,
                layer_name='theta',
                prior=prior,
                activation='linear',
                **priors_kwargs
            )(input_tensor)
            p = pm.Deterministic(
                'p',
                T.nnet.softmax(theta)
            )

            out = pm.Categorical(
                'y',
                p=p,
                observed=observed,
                total_size=total_size,
            )

        return lk_model

    def bernoulli_lk(self, shape_in, input_tensor, out_shape, observed,
                     total_size, prior, **priors_kwargs):
        """
        """
        with pm.Model() as lk_model:

            theta = Dense(
                shape_in=shape_in,
                units=out_shape,
                layer_name='theta',
                prior=prior,
                activation='linear',
                **priors_kwargs
            )(input_tensor)
            p = pm.Deterministic(
                'p',
                T.nnet.sigmoid(theta)
            )

            out = pm.Bernoulli(
                'y',
                p=p,
                observed=observed,
                total_size=total_size,
            )

        return lk_model
