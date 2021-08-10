import abc
from abc import ABC

import pymc3 as pm
import theano
import theano.tensor as tt

from .initializations import Initializations
from .activations import Activations


class _AbstractLayer(ABC, Initializations, Activations):
    """
    """
    def __init__(self):
        pass

    def __call__(self, input_tensor):
        """
        """
        output_tensor = self.build(
            input_tensor=input_tensor
        )
        return output_tensor

    @abc.abstractmethod
    def build(self):
        pass


class Dense(_AbstractLayer):
    """
    """
    def __init__(self, shape_in, units, layer_name, prior,
                 activation='sigmoid', weight_init_func='gaussian',
                 bias_init_func='gaussian', **priors_kwargs):
        """
        """
        self.shape = (shape_in, units) if units > 1 else (shape_in, )
        self.units = units
        self.layer_name = layer_name
        self.weight_init_func = getattr(self, weight_init_func)
        self.bias_init_func = getattr(self, bias_init_func)
        self.activation = getattr(self, activation)
        self.prior = prior
        self.priors_kwargs = priors_kwargs

    def __dense_init(self):
        """
        """
        floatX = theano.config.floatX

        weights_init = self.weight_init_func(
            shape=self.shape
        ).astype(floatX)

        biases_init = self.bias_init_func(
            shape=self.units
        )
        return weights_init, biases_init

    def __dense_weights(self, *args):
        """
        """
        weights_init, biases_init = args[0]

        weights = self.prior(
            f'dense_weights_{self.layer_name}',
            shape=self.shape,
            testval=weights_init,
            **self.priors_kwargs
        )
        biases = self.prior(
            f'dense_biases_{self.layer_name}',
            shape=self.units,
            testval=biases_init,
            **self.priors_kwargs
        )
        return weights, biases

    def build(self, input_tensor):
        """
        """
        weights, biases = self.__dense_weights(
            self.__dense_init()
        )
        output_tensor = pm.Deterministic(
                f'dense_{self.layer_name}',
                self.activation(
                    pm.math.dot(input_tensor, weights) + biases
                )
        )
        return output_tensor


class Embedding(_AbstractLayer):
    """
    """
    def __init__(self, vocabulary_size, units, layer_name, prior,
                 activation='sigmoid', embedding_init_func='gaussian',
                 **priors_kwargs):
        """
        """
        self.vocabulary_size = vocabulary_size
        self.units = units
        self.layer_name = layer_name
        self.embedding_init_func = getattr(self, embedding_init_func)
        self.activation = getattr(self, activation)
        self.prior = prior
        self.priors_kwargs = priors_kwargs

    def __embedding_init(self):
        """
        """
        floatX = theano.config.floatX
        weights_init = self.embedding_init_func(
            shape=(self.vocabulary_size, self.units)
        ).astype(floatX)
        return weights_init

    def __embedding_weights(self, weights_init):
        """
        """
        weights = self.prior(
            f'embedding_weights_{self.layer_name}',
            shape=(self.vocabulary_size, self.units),
            testval=weights_init,
            **self.priors_kwargs
        )
        return weights

    def build(self, input_tensor):
        """
        """
        weights = self.__embedding_weights(
            self.__embedding_init()
        )
        output_tensor = pm.Deterministic(
            f'embedding_{self.layer_name}',
            self.activation(
                weights
            )[input_tensor]
        )
        return output_tensor


class RNN(_AbstractLayer):
    """
    """
    def __init__(self, shapes, units, layer_name, prior,
                 return_sequences=False, weight_init_func='gaussian',
                 bias_init_func='gaussian', **priors_kwargs):
        """
        """
        self.units = units
        self.return_sequences = return_sequences

        # ugly shpe specification
        self.shape_batch = shapes[0]
        self.shape_seq = shapes[1]
        self.shape_feat = shapes[2]

        self.layer_name = layer_name
        self.weight_init_func = getattr(self, weight_init_func)
        self.bias_init_func = getattr(self, bias_init_func)
        self.prior = prior
        self.priors_kwargs = priors_kwargs

    def __RNN_init(self):
        """
        """
        floatX = theano.config.floatX

        iw_init = self.weight_init_func(
            shape=(self.shape_feat, self.units)
        ).astype(floatX)
        hw_init = self.weight_init_func(
            shape=(self.units, self.units)
        ).astype(floatX)

        hb_init = self.bias_init_func(
            shape=self.units
        )

        return iw_init, hw_init, hb_init

    def __RNN_weights(self, *args):
        """
        """
        iw_init, hw_init, hb_init = args[0]

        iw = self.prior(
            f'input_weights_{self.layer_name}',
            shape=(self.shape_feat, self.units),
            testval=iw_init,
            **self.priors_kwargs
        )
        hw = self.prior(
            f'hidden_weights_{self.layer_name}',
            shape=(self.units, self.units),
            testval=hw_init,
            **self.priors_kwargs
        )
        hb = self.prior(
            f'hidden_biases_{self.layer_name}',
            shape=self.units,
            testval=hb_init,
            **self.priors_kwargs
        )

        return iw, hw, hb

    def build(self, input_tensor):
        """
        """
        hidden_state = pm.Normal(
            'initial_hidden_state',
            mu=0,
            sigma=1,
            shape=(self.shape_batch, self.units)
        )
        hidden_states = [hidden_state]

        iw, hw, hb = self.__RNN_weights(
            self.__RNN_init()
        )

        for step in range(self.shape_seq):

            inp = pm.math.dot(input_tensor[:, step, :], iw)
            h = pm.math.dot(hidden_states[step], hw)
            a = hb + inp + h

            hidden_state = pm.Deterministic(
                f'{step}_hidden_state',
                pm.math.tanh(
                    a
                )
            )

            hidden_states.append(hidden_state)

        if self.return_sequences:
            raise NotImplementedError
        else:
            return hidden_states[-1]


class LSTM(_AbstractLayer):
    """
    """
    def __init__(self, shapes, units, layer_name, prior,
                 return_sequences=False, weight_init_func='gaussian',
                 bias_init_func='gaussian', **priors_kwargs):
        """
        """
        self.units = units
        self.return_sequences = return_sequences

        # ugly shpe specification
        self.shape_batch = shapes[0]
        self.shape_seq = shapes[1]
        self.shape_feat = shapes[2]
        self.shape_z = self.shape_feat + units

        self.layer_name = layer_name
        self.weight_init_func = getattr(self, weight_init_func)
        self.bias_init_func = getattr(self, bias_init_func)
        self.prior = prior
        self.priors_kwargs = priors_kwargs

    def __LSTM_init(self):
        """
        """
        floatX = theano.config.floatX

        fw_init = self.weight_init_func(
            shape=(self.shape_z, self.units)
        ).astype(floatX)
        fb_init = self.bias_init_func(
            shape=self.units
        )

        iw_init = self.weight_init_func(
            shape=(self.shape_z, self.units)
        ).astype(floatX)
        ib_init = self.bias_init_func(
            shape=self.units
        )

        ow_init = self.weight_init_func(
            shape=(self.shape_z, self.units)
        ).astype(floatX)
        ob_init = self.bias_init_func(
            shape=self.units
        )

        cw_init = self.weight_init_func(
            shape=(self.shape_z, self.units)
        ).astype(floatX)
        cb_init = self.bias_init_func(
            shape=self.units
        )

        return fw_init, fb_init, iw_init, ib_init, \
            ow_init, ob_init, cw_init, cb_init

    def __LSTM_weights(self, *args):
        """
        """
        fw_init, fb_init, iw_init, ib_init, ow_init, ob_init, cw_init, \
            cb_init = args[0]

        fw = self.prior(
            f'forget_weights_{self.layer_name}',
            shape=(self.shape_z, self.units),
            testval=fw_init,
            **self.priors_kwargs
        )
        fb = self.prior(
            f'forget_biases_{self.layer_name}',
            shape=self.units,
            testval=fb_init,
            **self.priors_kwargs
        )

        iw = self.prior(
            f'input_weights_{self.layer_name}',
            shape=(self.shape_z, self.units),
            testval=iw_init,
            **self.priors_kwargs
        )
        ib = self.prior(
            f'input_biases_{self.layer_name}',
            shape=self.units,
            testval=ib_init,
            **self.priors_kwargs
        )

        ow = self.prior(
            f'output_weights_{self.layer_name}',
            shape=(self.shape_z, self.units),
            testval=ow_init,
            **self.priors_kwargs
        )
        ob = self.prior(
            f'output_biases_{self.layer_name}',
            shape=self.units,
            testval=ob_init,
            **self.priors_kwargs
        )

        cw = self.prior(
            f'cell_weights_{self.layer_name}',
            shape=(self.shape_z, self.units),
            testval=cw_init,
            **self.priors_kwargs
        )
        cb = self.prior(
            f'cell_biases_{self.layer_name}',
            shape=self.units,
            testval=cb_init,
            **self.priors_kwargs
        )

        return fw, fb, iw, ib, ow, ob, cw, cb

    def build(self, input_tensor):
        """
        """
        hidden_states = []
        cell_states = []

        h = pm.Normal(
            'h',
            mu=0,
            sigma=1,
            shape=(self.shape_batch, self.units)
        )
        c = pm.Normal(
            'c',
            mu=0,
            sigma=1,
            shape=(self.shape_batch, self.units)
        )

        fw, fb, iw, ib, ow, ob, cw, cb = self.__LSTM_weights(
            self.__LSTM_init()
        )

        def cell_ops(input_tensor, h, c, fw, fb, iw, ib, ow, ob, cw, cb):
            """
            """
            fz = tt.concatenate([input_tensor, h], axis=-1)
            iz = theano.clone(fz)
            cz = theano.clone(fz)
            oz = theano.clone(fz)

            # forget
            forget_gate = pm.math.sigmoid(
                pm.math.dot(fz, fw) + fb
            )
            c *= forget_gate

            # input
            input_gate_1 = pm.math.sigmoid(
                pm.math.dot(iz, iw) + ib
            )
            input_gate_2 = pm.math.tanh(
                pm.math.dot(cz, cw) + cb
            )
            input_cell = input_gate_1 * input_gate_2
            c += input_cell

            # output
            output_gate = pm.math.sigmoid(
                pm.math.dot(oz, ow) + ob
            )
            co = pm.math.tanh(theano.clone(c))
            h = output_gate * co

            return h, c

        for i in range(self.shape_seq):

            h, c, = cell_ops(
                    input_tensor[:, i, :],
                    h=h,
                    c=c,
                    fw=fw,
                    fb=fb,
                    iw=iw,
                    ib=ib,
                    ow=ow,
                    ob=ob,
                    cw=cw,
                    cb=cb
                )

            hidden_states.append(h)
            cell_states.append(c)

        if self.return_sequences:
            raise NotImplementedError
        else:
            return hidden_states[-1], cell_states[-1]
