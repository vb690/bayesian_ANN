import pymc3 as pm

import theano.tensor as T


def normal_lk(previous_layer, weights, biases, observed, total_size,
              beta_cauchy=1):
    """
    """
    with pm.Model() as lk_model:
        mu = pm.Deterministic(
            'mu',
            pm.math.dot(
                previous_layer, weights
            ) + biases
        )
        sd = pm.HalfCauchy(
            'sd',
            beta=beta_cauchy
        )

        out = pm.Normal(
            'y',
            mu=mu,
            sd=sd,
            observed=observed,
            total_size=total_size,
        )

    return lk_model


def student_lk(previous_layer, weights, biases, observed, total_size,
               beta_cauchy=1, beta_gamma=0.1, alpha_gamma=2):
    """
    """
    with pm.Model() as lk_model:
        mu = pm.Deterministic(
            'mu',
            pm.math.dot(
                previous_layer, weights
            ) + biases
        )
        sd = pm.HalfCauchy(
            'sd',
            beta=beta_cauchy
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


def categorical_lk(previous_layer, weights, biases, observed, total_size):
    """
    """
    with pm.Model() as lk_model:
        p = pm.Deterministic(
            'p',
            T.nnet.softmax(
                pm.math.dot(
                    previous_layer, weights
                ) + biases
            )
        )

        out = pm.Categorical(
            'y',
            p=p,
            observed=observed,
            total_size=total_size,
        )

    return lk_model


def bernoulli_lk(previous_layer, weights, biases, observed, total_size):
    """
    """
    with pm.Model() as lk_model:
        p = pm.Deterministic(
            'p',
            T.nnet.sigmoid(
                pm.math.dot(
                    previous_layer, weights
                ) + biases
            )
        )

        out = pm.Bernoulli(
            'y',
            p=p,
            observed=observed,
            total_size=total_size,
        )

    return lk_model
