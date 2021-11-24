# Bayesian Artificial Neural Networks
Small project for exploring the usage of bayesian artificial neural networks implemented with PyMC3.

## Motivation

## Features

1. PymC3 implementations of various Artificial Neural Networks layers "à la Keras":

    * Dense Layer
    * Embedding Layer **(currently not working)**
    * Vanilla RNN Layer **(currently not working)**

2. Flexible pirors specification.
3. Set of pre-implemented Artificial Neural Networks with flexible likelyhood specification:
    
    * Multilayer Perceptron
    * Word Embedding **(currently not working)**

## Usage

###  Model Specification
Models can be specified stacking layers on top of each other and making sure to have a final likelyhood model.

```python
import numpy as np

import pymc3 as pm

from modules.layers import Dense

def gaussian_lk(shape_in, input_tensor, out_shape, observed,
                prior, beta=5, **priors_kwargs):
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
          observed=observed
      )

  return lk_model

X = np.random.random(size=(10000, 100))
y = np.random.random(size=(10000))
shape_in = dense.shape[1]

with pm.Model() as model:
   dense=X
   for layer_n, units in enumerate((100, 50, 25):

       dense = Dense(
           shape_in=shape_in,
           units=units,
           layer_name=layer_n,
           prior=pm.Normal,
           activation='relu',
           mu=0,
           sigma=1
       )(dense)
       shape_in = units
   
    out = likelyhood_model(
       shape_in=shape_in,
       input_tensor=dense,
       out_shape=1,
       observed=y,
       prior=pm.Normal,
       mu=0,
       sigma=1
   )
```

### Using pre-implemented models
A more conveninet shortcut is to use pre-implemented architectures like those specificed in the Features section.
```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedShuffleSplit

import pymc3 as pm

from modules.neural_networks import BayesianMLP


X, y = load_digits(return_X_y=True)

for tr_i, ts_i in StratifiedShuffleSplit(n_splits=1).split(X, y):
    
    X_tr, X_ts = X[tr_i], X[ts_i]
    y_tr, y_ts = y[tr_i], y[ts_i]

    scaler = MinMaxScaler()
    scaler.fit(X_tr)

    X_tr = scaler.transform(X_tr)
    X_ts = scaler.transform(X_ts)
    
categorical_perceptron = BayesianMLP(
    X=X_tr, 
    y=y_tr, 
    shape_out=10, 
    likelyhood_model='categorical_lk',
    layers=(128, 64, 32, 16), 
    activation='tanh',
    prior=pm.Normal,
    mu=0,
    sigma=0.1,
    batch_size=150
)
```
### Notebooks

1. Binary and multiclass classification.
2. Regression with Normal and Stundent T likelyhoods.
3. Sampling the embedding.
4. Language Model.
