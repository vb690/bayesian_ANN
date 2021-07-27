import pymc3 as pm

from .layers import Dense


class LikelyhoodModels:

    def __init__(self):
        """
        """
        pass

    def gaussian_lk(self, input_tensor, out_shape, observed, total_size, prior,
                    beta=25, **priors_kwargs):
        """
        """
        with pm.Model() as lk_model:

            mu = Dense(
                units=out_shape,
                layer_name='mu',
                prior=prior,
                activation='linear',
                **priors_kwargs
            )(input_tensor)

            sd = pm.HalfCauchy(
                name='sigma',
                beta=50
            )

            out = pm.Normal(
                'y',
                mu=mu,
                sd=sd,
                observed=observed,
                total_size=total_size,
            )

        return lk_model

    def student_lk(self, previous_layer, observed, total_size, beta_cauchy=25,
                   alpha_gamma=2, beta_gamma=0.1, **priors_kwargs):
        """
        """
        with pm.Model() as lk_model:

            mu = self.dense_layer(
                layer_name='mu',
                units=self.shape_out,
                shape_in=self.layers[-1],
                previous_layer=previous_layer,
                weight_init_func=self.weight_init_func,
                bias_init_func=self.bias_init_func,
                prior=self.prior,
                activation='linear',
                **priors_kwargs
            )
            sd = pm.HalfCauchy(
                name='sigma',
                beta=50
            )
            nu = pm.Gamma(
                'nu',
                alpha=alpha_gamma,
                beta=beta_gamma
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

    def categorical_lk(self, previous_layer, observed, total_size,
                       **priors_kwargs):
        """
        """
        with pm.Model() as lk_model:

            theta = self.dense_layer(
                layer_name='theta',
                units=self.shape_out,
                shape_in=self.layers[-1],
                previous_layer=previous_layer,
                weight_init_func=self.weight_init_func,
                bias_init_func=self.bias_init_func,
                prior=self.prior,
                activation='linear',
                **priors_kwargs
            )
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

    def bernoulli_lk(self, previous_layer, observed, total_size,
                     **priors_kwargs):
        """
        """
        with pm.Model() as lk_model:

            theta = self.dense_layer(
                layer_name='theta',
                units=self.shape_out,
                shape_in=self.layers[-1],
                previous_layer=previous_layer,
                weight_init_func=self.weight_init_func,
                bias_init_func=self.bias_init_func,
                prior=self.prior,
                activation='linear',
                **priors_kwargs
            )
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
