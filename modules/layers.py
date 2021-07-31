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
        self.shape_in = shape_in
        self.units = units
        self.layer_name = layer_name
        self.weight_init_func = getattr(self, weight_init_func)
        self.bias_init_func = getattr(self, bias_init_func)
        self.activation = getattr(self, activation)
        print(activation)
        self.prior = prior
        self.priors_kwargs = priors_kwargs

    def __dense_init(self):
        """
        """
        floatX = theano.config.floatX
        if self.units == 1:
            shape = (self.shape_in,)
        else:
            shape = (self.shape_in, self.units)

        weights_init = self.weight_init_func(
            shape=shape
        ).astype(floatX)

        biases_init = self.bias_init_func(
            shape=self.units
        )
        return weights_init, biases_init

    def __dense_weights(self, *args):
        """
        """
        weights_init, biases_init = args[0]
        if self.units == 1:
            shape = (self.shape_in,)
        else:
            shape = (self.shape_in, self.units)

        weights = self.prior(
            f'dense_weights_{self.layer_name}',
            shape=shape,
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
    def __init__(self, units, layer_name, prior, activation='sigmoid',
                 embedding_init_func='gaussian', **priors_kwargs):
        """
        """
        self.units = units
        self.layer_name = layer_name
        self.embedding_init_func = getattr(self, embedding_init_func)
        self.activation = getattr(self, activation)
        self.prior = prior
        self.priors_kwargs = priors_kwargs

    def __embedding_init(self, shape_in):
        """
        """
        floatX = theano.config.floatX
        weights_init = self.embedding_init_func(
            shape=(shape_in, self.units)
        ).astype(floatX)
        return weights_init

    def __embedding_weights(self, shape_in, embedding_init):
        """
        """
        weights = self.prior(
            f'embedding_weights_{self.layer_name}',
            shape=(shape_in, self.units),
            testval=embedding_init,
            **self.priors_kwargs
        )
        return weights

    def build(self, input_tensor):
        """
        """
        shape_in = input_tensor.shape.eval()[1]
        embedding_init = self.__embedding_init(
            shape_in=shape_in
        )
        weights = self.__embedding_weights(
            shape_in=shape_in,
            embedding_init=embedding_init
        )

        output_tensor = pm.Deterministic(
            f'embedding_{self.layer_name}',
            self.activation(
                weights
            )[self.input_tensor]
        )
        return output_tensor


class LSTM(_AbstractLayer):
    """
    """
    def __init__(self, units, layer_name, prior,
                 weight_init_func='gaussian', bias_init_func='gaussian',
                 **priors_kwargs):
        self.units = units
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

        return fw_init, fb_init, iw_init, ib_init,
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
        fw, fb, iw, ib, ow, ob, cw, cb = self.__LSTM_weights(
            self.__LSTM_init()
        )

        def cell_op(input_tensor, h, c, fw, fb, iw, ib, ow, ob, cw, cb):
            """
            """
            fz = pm.math.stack(input_tensor, h, axis=0)
            iz = theano.clone(fz)
            cz = theano.clone(fz)
            oz = theano.clone(fz)

            # forget
            forget_gate = pm.math.sigmoid(
                pm.dot(fz, fw) + fb
            )
            c *= forget_gate

            # input
            input_gate_1 = pm.math.sigmoid(
                pm.dot(iz, iw) + ib
            )
            input_gate_2 = pm.math.tanh(
                pm.dot(cz, cw) + cb
            )
            input_cell = input_gate_1 * input_gate_2
            c += input_cell

            # output
            output_gate = pm.math.sigmoid(
                pm.dot(oz, ow) + ob
            )
            co = pm.math.tanh(theano.clone(c))
            h = output_gate * co

            return h, c

        h = pm.Deterministic(
            'h',
            tt.zeros_like(input_tensor.shape[0], self.units)
        )
        c = pm.Deterministic(
            'c',
            tt.zeros_like(input_tensor.shape[0], self.units)
        )

        for i in range(self.sequence_len):

            h, c, = cell_op(
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

        return h, c
