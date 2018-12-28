from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from graph_nets import utils_tf
from graph_nets import utils_np
from graph_nets import graphs
from graph_nets import blocks
from graph_nets import modules
import sonnet as snt

ENCODER_NBR_LAYERS = 3
EMBEDDING_DIM = 128
HEAD_NBR = 8
KEY_DIM = 16
VALUE_DIM = 16
QUERY_DIM = 16
INIT_DIM = 2
INIT_MAX = 1 / np.sqrt(INIT_DIM)
FF_HIDDEN_SIZE = 512
N_NODE = 50
BATCH = 64
C = 10


class Decoder(snt.AbstractModule):

    def __init__(self, name="decoder-attention-tsp"):
        super(Decoder, self).__init__(name=name)

        with self._enter_variable_scope():
            initializers = {"w": tf.initializers.random_uniform(-INIT_DIM, INIT_DIM)}
            self._query_layer_0 = snt.Linear(output_size=HEAD_NBR * QUERY_DIM,
                                             use_bias=False,
                                             initializers=initializers,
                                             name="query_computer_0")
            self._value_layer_0 = snt.Linear(output_size=HEAD_NBR * VALUE_DIM,
                                             use_bias=False,
                                             initializers=initializers,
                                             name="value_computer_0")
            self._key_layer_0 = snt.Linear(output_size=HEAD_NBR * KEY_DIM,
                                           use_bias=False,
                                           initializers=initializers,
                                           name="key_computer_0")
            self._query_layer_1 = snt.Linear(output_size=EMBEDDING_DIM,
                                             use_bias=False,
                                             initializers=initializers,
                                             name="query_computer_1")
            self._key_layer_1 = snt.Linear(output_size=EMBEDDING_DIM,
                                           use_bias=False,
                                           initializers=initializers,
                                           name="key_computer_1")
            self._v1 = tf.get_variable(name="v1",
                                       shape=(1, EMBEDDING_DIM))
            self._vf = tf.get_variable(name="vf",
                                       shape=(1, EMBEDDING_DIM))
            self._W = tf.get_variable(name="multi_head_reducer",
                                      shape=(VALUE_DIM, EMBEDDING_DIM),
                                      initializer=tf.initializers.random_uniform(-INIT_DIM, INIT_DIM))

    def _build(self, graph):
        """Decodes the graph

        Args:
            graph (graphs.GraphsTuple):

        Returns:
            The ordered indices
        """

        # assert tf.reduce_all(tf.equal(graph.n_node[0], graph.n_node)), "Not all the graphs have the same size!"

        nodes = graph.nodes
        graph_em = graph.globals

        value = tf.reshape(self._value_layer_0(nodes), (BATCH, N_NODE, HEAD_NBR, VALUE_DIM))
        key = tf.reshape(self._key_layer_0(nodes), (BATCH, N_NODE, HEAD_NBR, KEY_DIM))
        key_1 = tf.reshape(self._key_layer_1(nodes), (BATCH, N_NODE, EMBEDDING_DIM))

        nodes = tf.reshape(nodes, (BATCH, N_NODE, EMBEDDING_DIM))
        # pi = tf.Variable(initial_value=np.zeros((BATCH, N_NODE), dtype=np.int64), trainable=False, dtype=tf.int64)
        pi = tf.get_variable(name="pi",
                             initializer=tf.convert_to_tensor(np.full((BATCH, N_NODE), 0, dtype=np.int64)),
                             dtype=tf.int64,
                             trainable=False)

        # mask = tf.Variable(initial_value=np.full((BATCH, N_NODE), True), trainable=False, dtype=tf.bool)
        mask = tf.get_variable(name="mask",
                               initializer=tf.convert_to_tensor(np.full((BATCH, N_NODE), True)),
                               dtype=tf.bool,
                               trainable=False)

        mask.assign(tf.convert_to_tensor(np.full((BATCH, N_NODE), True)))
        pi.assign(tf.convert_to_tensor(np.full((BATCH, N_NODE), 0, dtype=np.int64)))

        log_p = None

        for t in range(N_NODE):
            if t == 0:
                v1 = tf.tile(self._v1, tf.convert_to_tensor([BATCH, 1]))
                vf = tf.tile(self._vf, tf.convert_to_tensor([BATCH, 1]))
                h_c = tf.concat([graph_em, v1, vf], axis=1)
            else:

                h1 = tf.gather_nd(nodes, tf.stack([tf.range(BATCH, dtype=tf.int64), pi[:, 0]], axis=1))
                hf = tf.gather_nd(nodes, tf.stack([tf.range(BATCH, dtype=tf.int64), pi[:, t - 1]], axis=1))
                h_c = tf.concat([graph_em, hf, h1], axis=1)  # (BATCH, 3*EMBEDDING_DIM)

            query_context = tf.reshape(self._query_layer_0(h_c), (BATCH, 1, HEAD_NBR, QUERY_DIM))

            if t != 0:
                mask = tf.scatter_nd_update(mask,
                                            tf.stack([tf.range(BATCH, dtype=tf.int64), pi[:, t - 1]], axis=1),
                                            tf.broadcast_to(False, [BATCH]))

            masked_key = tf.boolean_mask(key, mask)

            u_c = tf.multiply(query_context, key)
            u_c = tf.reduce_sum(u_c / tf.convert_to_tensor(np.sqrt(KEY_DIM).astype(dtype=np.float32)), axis=-1)
            u_c = tf.boolean_mask(u_c, mask)
            a = tf.reshape(tf.nn.softmax(u_c), (BATCH, N_NODE - t, HEAD_NBR, 1))  # (BATCH, ?, HEAD_NBR)

            masked_value = tf.reshape(tf.boolean_mask(value, mask), (BATCH, N_NODE - t, HEAD_NBR, VALUE_DIM))
            h_tmp = tf.reduce_sum(tf.multiply(masked_value, a), axis=1)  # (BATCH, HEAD_NBR, VALUE_DIM)
            h_c_n = tf.reduce_sum(tf.einsum('ijk,kl->ijl', h_tmp, self._W), axis=1)  # (BATCH, EMBEDDING_DIM)

            query_c = tf.expand_dims(self._query_layer_1(h_c_n), axis=1)  # (BATCH, 1, EMBEDDING_DIM)

            u_c_n = C * tf.nn.tanh(
                tf.reduce_sum(tf.multiply(query_c, key_1), axis=2) / tf.convert_to_tensor(
                    np.sqrt(EMBEDDING_DIM).astype(np.float32)))
            # u_c_n.shape = (BATCH, N_NODE)

            if t != 0:
                indices = tf.cast(tf.where(tf.logical_not(mask)), tf.int64)
                print('indices', indices)
                val_masked = tf.scatter_nd(indices, tf.broadcast_to(tf.float32.min, [N_NODE * t]),
                                           shape=(BATCH, N_NODE))
                u_c_n = tf.add(u_c_n, val_masked)

            proba = tf.nn.log_softmax(u_c_n, axis=1)

            pi_t = tf.argmax(proba, axis=1)  # ?

            pi = tf.scatter_nd_update(pi,
                                      tf.stack([tf.range(BATCH, dtype=tf.int64),
                                                tf.cast(tf.broadcast_to(t, [BATCH]), tf.int64)], axis=1),
                                      pi_t)

            if t == 0:
                log_p = tf.reduce_max(proba, axis=1)
            else:
                log_p = tf.add(log_p, tf.reduce_max(proba, axis=1))

        return pi, log_p
