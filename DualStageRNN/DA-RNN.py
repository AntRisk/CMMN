# encoding:utf-8

import numpy as np
import tensorflow as tf
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


class InputAttnCell(RNNCell):
    def __init__(self,
                 num_units,
                 u_e_x,
                 max_time,
                 forget_bias=1.0,
                 activation=None,
                 reuse=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(InputAttnCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._forget_bias = forget_bias
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh

        self._u_e_x = u_e_x
        self._max_time = max_time

    @property
    def state_size(self):
        return 2 * self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        self._input_depth = inputs_shape[-1]
        h_depth = self._num_units
        self._kernel = self.add_variable(
            "ia_kernel",
            shape=[self._input_depth + h_depth, 4 * self._num_units])
        self._bias = self.add_variable(
            "ia_bias",
            shape=[4 * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self._v_e = self.add_variable(
            "ia_v_e",
            shape=[self._max_time])
        self._w_e = self.add_variable(
            "ia_w_e",
            shape=[2 * self._num_units, self._max_time])
        self.built = True

    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)

        add = math_ops.add
        multiply = math_ops.multiply

        e_k = math_ops.matmul(state, self._w_e)
        e_k = tf.tile(e_k, [1, self._input_depth])
        e_k = tf.reshape(e_k, [-1, self._input_depth, self._max_time])
        e_k = add(e_k, self._u_e_x)
        e_k = math_ops.tanh(e_k)
        e_k = tf.tensordot(e_k, self._v_e, axes=[[2], [0]])
        a_t = tf.nn.softmax(e_k, axis=1)
        inputs = multiply(inputs, a_t)

        c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, h], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        i, j, f, o = array_ops.split(
            value=gate_inputs, num_or_size_splits=4, axis=one)

        forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                    multiply(sigmoid(i), self._activation(j)))
        new_h = multiply(self._activation(new_c), sigmoid(o))

        new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state

    def get_config(self):
        config = {
            "num_units": self._num_units,
            "forget_bias": self._forget_bias,
            "state_is_tuple": self._state_is_tuple,
            "activation": activations.serialize(self._activation),
            "reuse": self._reuse,
        }
        base_config = super(InputAttnCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TemporalAttnCell(RNNCell):
    def __init__(self,
                 num_units,
                 all_inputs,
                 u_d_h,
                 forget_bias=1.0,
                 activation=None,
                 reuse=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(TemporalAttnCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs)

        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._forget_bias = forget_bias
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh

        self._h = all_inputs
        self._h_depth = all_inputs.get_shape()[-1]
        self._h_length = all_inputs.get_shape()[-2]
        self._u_d_h = u_d_h

    @property
    def state_size(self):
        return 2 * self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)
        self._y_depth = inputs_shape[-1]
        d_depth = self._num_units

        self._kernel = self.add_variable(
            "ta_kernel",
            shape=[self._y_depth + d_depth, 4 * d_depth])
        self._bias = self.add_variable(
            "ta_bias",
            shape=[4 * d_depth],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self._v_d = self.add_variable(
            "ta_v_d",
            shape=[self._h_depth])
        self._w_d = self.add_variable(
            "ta_w_d",
            shape=[2 * d_depth, self._h_depth])

        self._w_tilde = self.add_variable(
            "ta_w_tiled",
            shape=[self._h_depth + self._y_depth, 1]
        )
        self._b_tilde = self.add_variable(
            "ta_b_tiled",
            shape=[1],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def get_attn(self, state):
        # temporal attn
        l_t = math_ops.matmul(state, self._w_d)
        l_t = tf.tile(l_t, [1, self._h_length])
        l_t = tf.reshape(l_t, [-1, self._h_length, self._h_depth])
        l_t = math_ops.add(l_t, self._u_d_h)
        l_t = math_ops.tanh(l_t)
        l_t = tf.tensordot(l_t, self._v_d, axes=[[2], [0]])
        b_t = tf.nn.softmax(l_t, axis=1)
        b_t = tf.tile(b_t, [1, self._h_depth])
        b_t = tf.reshape(b_t, [-1, self._h_length, self._h_depth])
        c_t_1 = tf.reduce_sum(math_ops.multiply(b_t, self._h), axis=1)
        return c_t_1

    def call(self, y, state):
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)

        add = math_ops.add
        multiply = math_ops.multiply

        attn = self.get_attn(state)

        y_tilde = math_ops.matmul(
            array_ops.concat([attn, y], 1), self._w_tilde)
        y_tilde = nn_ops.bias_add(y_tilde, self._b_tilde)

        s, d = array_ops.split(value=state, num_or_size_splits=2, axis=one)

        gate_inputs = math_ops.matmul(
            array_ops.concat([y_tilde, d], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        i, j, f, o = array_ops.split(
            value=gate_inputs, num_or_size_splits=4, axis=one)

        forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
        new_s = add(multiply(s, sigmoid(add(f, forget_bias_tensor))),
                    multiply(sigmoid(i), self._activation(j)))
        new_d = multiply(self._activation(new_s), sigmoid(o))

        new_state = array_ops.concat([new_s, new_d], 1)
        return new_d, new_state

    def get_config(self):
        config = {
            "num_units": self._num_units,
            "forget_bias": self._forget_bias,
            "state_is_tuple": self._state_is_tuple,
            "activation": activations.serialize(self._activation),
            "reuse": self._reuse,
        }
        base_config = super(TemporalAttnCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def fill_data(data):
    length = [np.shape(d)[0] for d in data[0]]

    depth = [np.shape(d[0])[1] for d in data]

    max_time = np.max(length)

    res_data = []
    for type, dep in zip(data, depth):
        tmp_t = []
        for d in type:
            d += [[np.NaN for j in range(dep)] for i in range(max_time - len(d))]
            tmp_t.append(d)
        res_data.append(np.array(tmp_t))
    return res_data, length, depth


def DALSTM(X, Y, x_max_time, y_max_time, h_depth, d_depth):
    X_T = tf.transpose(X, [0, 2, 1])

    Ue = tf.Variable(tf.truncated_normal([x_max_time, x_max_time], dtype=tf.float32))
    UeX = tf.tensordot(X_T, Ue, axes=[[2], [0]])
    input_cell = InputAttnCell(h_depth, UeX, x_max_time)
    H, _ = tf.nn.dynamic_rnn(input_cell, X, dtype=tf.float32)

    Ud = tf.Variable(tf.truncated_normal([h_depth, h_depth], dtype=tf.float32))
    UdH = tf.tensordot(H, Ud, axes=[[2], [0]])
    temporal_cell = TemporalAttnCell(d_depth, H, UdH)
    outputs, state = tf.nn.dynamic_rnn(temporal_cell, Y, dtype=tf.float32)
    c_t = temporal_cell.get_attn(state)
    _, d_t = array_ops.split(value=state, num_or_size_splits=2, axis=1)

    Wy = tf.Variable(tf.truncated_normal([h_depth + d_depth, d_depth], dtype=tf.float32))
    bw = tf.Variable(tf.zeros([d_depth], dtype=tf.float32))
    Vy = tf.Variable(tf.truncated_normal([d_depth, 1], dtype=tf.float32))
    bv = tf.Variable(tf.zeros([1], dtype=tf.float32))
    y_hat = math_ops.matmul(array_ops.concat([d_t, c_t], axis=1), Wy)
    y_hat = math_ops.add(y_hat, bw)
    y_hat = math_ops.matmul(y_hat, Vy)
    y_hat = math_ops.add(y_hat, bv)

    return y_hat


def training_data():
    base = np.linspace(0, 10000, 100000)

    x_1 = np.cos(base)
    x_2 = np.sin(base)
    x = np.c_[x_1, x_2]

    y = np.max(x, axis=1) * 10
    return x, y


def sample(x, y, length, nums=100):
    sample_index = np.random.choice(range(0, len(y) - length - 1), nums)

    x_samples = []
    y_samples = []
    labels = []

    for i in sample_index:
        x_samples.append(x[i:i+length])
        y_samples.append(y[i:i+length])
        labels.append(y[i+length])

    x_samples = np.reshape(x_samples, [nums, length, -1])
    y_samples = np.reshape(y_samples, [nums, length, -1])
    labels = np.reshape(labels, [nums, -1])

    return x_samples, y_samples, labels


def testing_data(x, y, length):
    x_samples = []
    y_samples = []
    for i in range(1000):
        x_samples.append(x[i: i+length])
        y_samples.append(y[i: i+length])

    x_samples = np.array(x_samples)
    y_samples = np.reshape(np.array(y_samples), [-1, length, 1])

    return x_samples, y_samples


def data_test_darnn():
    max_time = 50
    xs, ys = training_data()

    x_max_time = max_time
    y_max_time = max_time
    x_depth = 2
    y_depth = 1
    h_depth = 50
    d_depth = 100

    X = tf.placeholder(tf.float32, shape=[None, x_max_time, x_depth])
    Y = tf.placeholder(tf.float32, shape=[None, y_max_time, y_depth])
    Y_hat = tf.placeholder(tf.float32, shape=[None, y_depth])
    prediction = DALSTM(X, Y, x_max_time, y_depth, h_depth, d_depth)

    loss = tf.losses.mean_squared_error(Y_hat, prediction)
    train_step = tf.train.AdamOptimizer().minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            x, y, labels = sample(xs, ys, max_time)
            l, _ = sess.run([loss, train_step], {X: x, Y: y, Y_hat: labels})
            print(l)

        x, y = testing_data(xs, ys, max_time)
        p = sess.run([prediction], {X: x, Y: y, Y_hat: labels})
        p = np.reshape(p, [-1, 1])
        plt.plot(p, c='b')
        plt.plot(ys[50:1051], c='g')
        plt.show()


def AT_LSTM(X, LEN, Y, KEEP_PROB):
    num_units = 50
    with tf.variable_scope("AT_LSTM") as variable_scope:
        cells = tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=False)
        cells = tf.nn.rnn_cell.DropoutWrapper(cells, output_keep_prob=KEEP_PROB)
        all_rnn_output, final_state = tf.nn.dynamic_rnn(cells, X, sequence_length=LEN, dtype=tf.float32)
        _, final_rnn_output = array_ops.split(value=final_state, num_or_size_splits=2, axis=1)
        W_a = tf.Variable(tf.random_normal([num_units, 1], stddev=0.35), name="attn_weights")
        attn = tf.tensordot(all_rnn_output, W_a, axes=[[2], [0]])
        attn = tf.nn.tanh(attn)
        attn = tf.nn.softmax(attn, axis=1)
        r = tf.reduce_sum(all_rnn_output * attn, axis=1)
        W = tf.Variable(tf.random_normal([num_units, 2], stddev=0.35), name="final_weights")
        b = tf.Variable(tf.zeros([2]), name="final_bias")
        logist = tf.matmul(r, W) + b
        softmax_logist = tf.nn.softmax(logist)
        final_output = tf.argmax(logist, axis=-1)

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logist)
        )
    return final_rnn_output, softmax_logist, final_output, loss


if __name__ == '__main__':
    data_test_darnn()

