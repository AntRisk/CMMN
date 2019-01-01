import random
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.layers import base as base_layer
from tensorflow.python.keras import activations
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes


MAX_TIME = 3
NUM_UNITS = 5
FEA_DIMS = 8


class TimeIntervalLSTMCell(RNNCell):
    """The implementation of TimeIntervalLSTM
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 name=None,
                 dtype=None,
                 is_ut=False,
                 **kwargs):
        super(TimeIntervalLSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)

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
            shape=[input_depth + self._num_units, 3 * self._num_units],
            initializer=initialiser)

        self._bias = self.add_variable(
            "bias",
            shape=[3 * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self._time_kernel = self.add_variable(
            "time_kernel",
            shape=[input_depth, 2 * self._num_units],
            initializer=initialiser)

        self._delta_time_kernel = self.add_variable(
            "delta_time_kernel",
            shape=[1, 3 * self._num_units],
            initializer=initialiser)

        self._x_kernel = self.add_variable(
            "x_kernel",
            shape=[input_depth, 2 * self._num_units],
            initializer=initialiser)

        self._time_bias = self.add_variable(
            "time_bias",
            shape=[2 * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid
        add = math_ops.add
        sub = math_ops.subtract
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

        i, j, o = array_ops.split(value=gate_inputs, num_or_size_splits=3, axis=one)
        t1, t2, t3 = array_ops.split(value=time_inputs, num_or_size_splits=3, axis=one)

        i_m = sigmoid(i)

        x_m = math_ops.matmul(x, self._x_kernel)
        x_m = nn_ops.bias_add(x_m, self._time_bias)
        x_m1, x_m2 = array_ops.split(value=x_m, num_or_size_splits=2, axis=one)

        t_m1 = sigmoid(add(x_m1, self._activation(t1)))
        t_m2 = sigmoid(add(x_m2, self._activation(t2)))

        c_m_tilde = add(multiply(c, sub(1.0, multiply(i_m, t_m1))), multiply(multiply(self._activation(j), i_m), t_m1))
        c_m = add(multiply(c, sub(1.0, i_m)), multiply(multiply(self._activation(j), i_m), t_m2))

        o_m = sigmoid(add(o, t3))

        h_m = multiply(o_m, self._activation(c_m_tilde))

        new_state = array_ops.concat([c_m, h_m], 1)
        return h_m, new_state

    def get_config(self):
        config = {
            "num_units": self._num_units,
            "activation": activations.serialize(self._activation),
            "reuse": self._reuse,
        }
        base_config = super(TimeIntervalLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def model_feabeh_timelstm3_fa_lstm_ta(X_1, LEN, KEEP_PROB, scope='rnn'):
    max_time = MAX_TIME
    num_units = NUM_UNITS
    with tf.variable_scope(scope + "1"):
        cells_1 = TimeIntervalLSTMCell(num_units)
        rnn_all_h_1, rnn_final_sh_1 = tf.nn.dynamic_rnn(cells_1, X_1, sequence_length=LEN, dtype=tf.float32)

        rnn_all_h_1_t = tf.transpose(rnn_all_h_1, [0, 2, 1])

        U_fa = tf.Variable(tf.random_normal([max_time, max_time], stddev=0.5))
        u_h = tf.tensordot(rnn_all_h_1_t, U_fa, axes=[[2], [0]])
        u_h = tf.transpose(u_h, [0, 2, 1])

        W_fa = tf.Variable(tf.random_normal([num_units, max_time], stddev=0.5))
        w_hs = tf.tensordot(rnn_all_h_1, W_fa, axes=[[2], [0]])
        w_hs = tf.reshape(tf.tile(w_hs, [1, 1, num_units]), [-1, max_time, max_time, num_units])
        w_hs = tf.transpose(w_hs, [1, 0, 2, 3])
        f_attn = tf.transpose(w_hs + u_h, [0, 1, 3, 2])
        f_attn = tf.nn.tanh(tf.transpose(f_attn, [1, 0, 2, 3]))

        V_fa = tf.Variable(tf.random_normal([max_time, 1], stddev=0.5))
        f_attn = tf.tensordot(f_attn, V_fa, axes=[[3], [0]])
        f_attn = tf.reduce_sum(f_attn, axis=-1)
        f_attn = tf.nn.softmax(f_attn, axis=-1)

    with tf.variable_scope(scope + "2"):
        X_2 = rnn_all_h_1 * f_attn
        cells_2 = tf.nn.rnn_cell.LSTMCell(num_units)
        rnn_all_h_2, rnn_final_sh_2 = tf.nn.dynamic_rnn(cells_2, X_2, sequence_length=LEN, dtype=tf.float32)
        (_, rnn_final_h_2) = rnn_final_sh_2

        W_ta = tf.Variable(tf.random_normal([num_units, 1], stddev=0.5))
        t_attn = tf.tensordot(rnn_all_h_2, W_ta, axes=[[2], [0]])
        t_attn = tf.nn.tanh(t_attn)
        t_attn = tf.nn.softmax(t_attn, axis=1)
        r = tf.reduce_sum(rnn_all_h_2 * t_attn, axis=1)

        fc_w_1 = tf.Variable(tf.random_normal([num_units, 10]))
        fc_b_1 = tf.Variable(tf.zeros([1, 10]) + 0.1)
        logits = tf.nn.relu(tf.matmul(r, fc_w_1) + fc_b_1)
        logits = tf.nn.dropout(logits, KEEP_PROB)
        fc_w_2 = tf.Variable(tf.random_normal([10, 2]))
        fc_b_2 = tf.Variable(tf.zeros([1, 2]) + 0.1)
        logits = tf.matmul(logits, fc_w_2) + fc_b_2

        softmax_logits = tf.nn.softmax(logits)
    return softmax_logits


def test_data():
    feature = np.array([[
        [0 for i in range(FEA_DIMS-1)],
        [1 for i in range(FEA_DIMS-1)],
        [random.random() for i in range(FEA_DIMS-1)]]] * 4)
    delta_time = np.array([[[0], [1], [random.random()]]] * 4)
    return np.concatenate((feature, delta_time), axis=2)


if __name__ == "__main__":
    data = test_data()
    X = tf.placeholder(dtype=tf.float32, shape=[None, MAX_TIME, FEA_DIMS])
    LEN = tf.placeholder(dtype=tf.float32, shape=[None])
    KEEP_PROB = tf.placeholder(dtype=tf.float32)

    out = model_feabeh_timelstm3_fa_lstm_ta(X, LEN, KEEP_PROB)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(out, feed_dict={X: data, LEN: [3, 3, 3, 3], KEEP_PROB: 1.0})
        print(res)
