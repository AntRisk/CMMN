# encoding:utf-8

import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.layers import base as base_layer
from tensorflow.python.keras import activations
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes


class TimeLSTM1Cell(RNNCell):
    """ An implementation of TimeLSTM1
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 name=None,
                 dtype=None,
                 is_ut=False,
                 **kwargs):
        super(TimeLSTM1Cell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)

        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh

        self._is_ut = is_ut

    @property
    def state_size(self):
        return 2 * self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        initialiser = None
        if self._is_ut:
            initialiser = init_ops.ones_initializer(dtype=self.dtype)

        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[-1] - 1

        self._kernel = self.add_variable(
            "kernel",
            shape=[input_depth + self._num_units, 4 * self._num_units],
            initializer=initialiser)

        self._in_cell_kernel = self.add_variable(
            "in_cell_kernel",
            shape=[self._num_units],
            initializer=initialiser)
        self._forget_cell_kernel = self.add_variable(
            "forget_cell_kernel",
            shape=[self._num_units],
            initializer=initialiser)
        self._out_cell_kernel = self.add_variable(
            "out_cell_kernel",
            shape=[self._num_units],
            initializer=initialiser)

        self._bias = self.add_variable(
            "bias",
            shape=[4 * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self._time_kernel = self.add_variable(
            "time_kernel",
            shape=[input_depth, self._num_units],
            initializer=initialiser)

        self._delta_time_kernel = self.add_variable(
            "delta_time_kernel",
            shape=[1, 2 * self._num_units],
            initializer=initialiser)

        self._x_kernel = self.add_variable(
            "x_kernel",
            shape=[input_depth, self._num_units],
            initializer=initialiser)

        self._time_bias = self.add_variable(
            "time_bias",
            shape=[self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid
        add = math_ops.add
        multiply = math_ops.multiply
        one = constant_op.constant(1, dtype=dtypes.int32)

        c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)
        x = inputs[:, :-1]
        delta_time = inputs[:, -1:]

        gate_inputs = math_ops.matmul(
            array_ops.concat([x, h], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        time_inputs = math_ops.matmul(
            delta_time, self._delta_time_kernel)

        i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=one)
        t1, t2 = array_ops.split(value=time_inputs, num_or_size_splits=2, axis=one)

        gate_cell_in = multiply(c, self._in_cell_kernel)
        i_m = sigmoid(add(i, gate_cell_in))

        gate_cell_forget = multiply(c, self._forget_cell_kernel)
        f_m = sigmoid(add(f, gate_cell_forget))

        x_m = math_ops.matmul(x, self._x_kernel)
        t_m = sigmoid(nn_ops.bias_add(
            add(x_m, self._activation(t1)),
            self._time_bias)
        )

        c_m = add(multiply(c, f_m), multiply(multiply(self._activation(j), i_m), t_m))

        gate_cell_out = multiply(c_m, self._out_cell_kernel)
        o_m = sigmoid(add(add(o, gate_cell_out), t2))

        h_m = multiply(o_m, self._activation(c_m))

        new_state = array_ops.concat([c_m, h_m], 1)
        return h_m, new_state

    def get_config(self):
        config = {
            "num_units": self._num_units,
            "activation": activations.serialize(self._activation),
            "reuse": self._reuse,
        }
        base_config = super(TimeLSTM1Cell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def TimeLSTM1(x, is_ut=False):
    rnn_cell = TimeLSTM1Cell(10, is_ut=is_ut)
    output, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
    return output, states


def test_data():
    # batch_size, sequence_len, vector_size
    feature = np.array([[
        [0 for i in range(8)],
        [1 for i in range(8)],
        [random.random() for i in range(8)]]] * 4)
    delta_time = np.array([[[0], [1], [random.random()]]] * 4)
    return np.concatenate((feature, delta_time), axis=2)


def target_net(data, num_units):
    data = np.transpose(data, [1, 0, 2])
    input_size = np.shape(data)[-1] - 1
    batch_size = np.shape(data)[1]

    h = np.zeros((batch_size, num_units))
    c = np.zeros((batch_size, num_units))

    w_xi = np.ones([input_size, num_units])
    w_hi = np.ones([num_units, num_units])
    w_ci = np.ones(num_units)
    b_i = np.zeros(num_units)

    w_xf = np.ones([input_size, num_units])
    w_hf = np.ones([num_units, num_units])
    w_cf = np.ones(num_units)
    b_f = np.zeros(num_units)

    w_xt = np.ones([input_size, num_units])
    w_tt = np.ones([1, num_units])
    b_t = np.zeros(num_units)

    w_xc = np.ones([input_size, num_units])
    w_hc = np.ones([num_units, num_units])
    b_c = np.zeros((batch_size, num_units))

    w_xo = np.ones([input_size, num_units])
    w_to = np.ones([1, num_units])
    w_ho = np.ones([num_units, num_units])
    w_co = np.ones(num_units)
    b_o = np.zeros((batch_size, num_units))

    sigmoid = lambda x: 1.0 / (1 + np.exp(-x))
    res = []

    for d in data:
        x = d[:, :-1]
        t = d[:, -1:]

        i_m = sigmoid(np.matmul(x, w_xi) + np.matmul(h, w_hi) + w_ci * c + b_i)
        f_m = sigmoid(np.matmul(x, w_xf) + np.matmul(h, w_hf) + w_cf * c + b_f)

        t_m = sigmoid(np.matmul(x, w_xt) + np.tanh(np.matmul(t, w_tt)) + b_t)

        c_m = f_m * c + i_m * t_m * np.tanh(np.matmul(x, w_xc) + np.matmul(h, w_hc) + b_c)
        o_m = sigmoid(np.matmul(x, w_xo) + np.matmul(t, w_to) + np.matmul(h, w_ho) + w_co * c_m + b_o)
        h_m = o_m * np.tanh(c_m)
        c = c_m
        h = h_m
        res.append(h_m)
    res = np.array(res).transpose([1, 0, 2])
    return res


def ut():
    x = test_data()
    target = target_net(x, 10)

    X = tf.placeholder(dtype=tf.float32, shape=[None, 3, 9], name="inputx")
    output, states = TimeLSTM1(X, True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        predict = sess.run(output, feed_dict={X: x})
        if np.abs(np.mean(target - predict)) < 0.0000001:
            print("Accept")
        else:
            print("Fail")


if __name__ == '__main__':
    ut()
