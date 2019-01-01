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


class Decoder(snt.AbstractModule):

    def __init__(self, conf, name="decoder-attention-tsp"):
        super(Decoder, self).__init__(name=name)
        self.conf = conf
        with self._enter_variable_scope():
            initializers = {"w": tf.initializers.random_uniform(-self.conf.init_max, self.conf.init_max)}
            self._query_layer_0 = snt.Linear(output_size=self.conf.head_nbr * self.conf.query_dim,
                                             use_bias=False,
                                             initializers=initializers,
                                             name="query_computer_0")
            self._value_layer_0 = snt.Linear(output_size=self.conf.head_nbr * self.conf.value_dim,
                                             use_bias=False,
                                             initializers=initializers,
                                             name="value_computer_0")
            self._key_layer_0 = snt.Linear(output_size=self.conf.head_nbr * self.conf.key_dim,
                                           use_bias=False,
                                           initializers=initializers,
                                           name="key_computer_0")
            self._query_layer_1 = snt.Linear(output_size=self.conf.embedding_dim,
                                             use_bias=False,
                                             initializers=initializers,
                                             name="query_computer_1")
            self._key_layer_1 = snt.Linear(output_size=self.conf.embedding_dim,
                                           use_bias=False,
                                           initializers=initializers,
                                           name="key_computer_1")
            self._v1 = tf.get_variable(name="v1",
                                       shape=(1, self.conf.embedding_dim))
            self._vf = tf.get_variable(name="vf",
                                       shape=(1, self.conf.embedding_dim))
            self._W = tf.get_variable(name="multi_head_reducer",
                                      shape=(self.conf.value_dim, self.conf.embedding_dim),
                                      initializer=tf.initializers.random_uniform(-self.conf.init_max,
                                                                                 self.conf.init_max))

    def _build(self, graph):
        """Decodes the graph

        Args:
            graph (graphs.GraphsTuple):

        Returns:
            The ordered indices
        """

        nodes = graph.nodes
        graph_em = graph.globals

        value = tf.reshape(self._value_layer_0(nodes),
                           (self.conf.batch, self.conf.n_node, self.conf.head_nbr, self.conf.value_dim))
        key = tf.reshape(self._key_layer_0(nodes),
                         (self.conf.batch, self.conf.n_node, self.conf.head_nbr, self.conf.key_dim))
        key_1 = tf.reshape(self._key_layer_1(nodes), (self.conf.batch, self.conf.n_node, self.conf.embedding_dim))

        nodes = tf.reshape(nodes, (self.conf.batch, self.conf.n_node, self.conf.embedding_dim))

        pi = tf.get_local_variable(name="pi",
                                   initializer=tf.convert_to_tensor(
                                       np.full((self.conf.batch, self.conf.n_node), 0, dtype=np.int64)),
                                   dtype=tf.int64,
                                   trainable=False)

        mask = tf.get_local_variable(name="mask",
                                     initializer=tf.convert_to_tensor(
                                         np.full((self.conf.batch, self.conf.n_node), True)),
                                     dtype=tf.bool,
                                     trainable=False)

        output_log_proba = []

        for t in range(self.conf.n_node):
            # Computing query context
            if t == 0:
                v1 = tf.tile(self._v1, tf.convert_to_tensor([self.conf.batch, 1]))
                vf = tf.tile(self._vf, tf.convert_to_tensor([self.conf.batch, 1]))
                h_c = tf.concat([graph_em, v1, vf], axis=1)
            else:

                h1 = tf.gather_nd(nodes, tf.stack([tf.range(self.conf.batch, dtype=tf.int64), pi[:, 0]], axis=1))
                hf = tf.gather_nd(nodes, tf.stack([tf.range(self.conf.batch, dtype=tf.int64), pi[:, t - 1]], axis=1))
                h_c = tf.concat([graph_em, hf, h1], axis=1)

            query_context = tf.reshape(self._query_layer_0(h_c),
                                       (self.conf.batch, 1, self.conf.head_nbr, self.conf.query_dim))

            if t != 0:
                mask = tf.scatter_nd_update(mask,
                                            tf.stack([tf.range(self.conf.batch, dtype=tf.int64), pi[:, t - 1]], axis=1),
                                            tf.broadcast_to(False, [self.conf.batch]))

            u_c = tf.multiply(query_context, key)
            u_c = tf.reduce_sum(u_c / tf.convert_to_tensor(np.sqrt(self.conf.key_dim).astype(dtype=np.float32)),
                                axis=-1)
            u_c = tf.boolean_mask(u_c, mask)
            a = tf.reshape(tf.nn.softmax(u_c), (self.conf.batch, self.conf.n_node - t, self.conf.head_nbr, 1))
            masked_value = tf.reshape(tf.boolean_mask(value, mask),
                                      (self.conf.batch, self.conf.n_node - t, self.conf.head_nbr, self.conf.value_dim))
            h_tmp = tf.reduce_sum(tf.multiply(masked_value, a), axis=1)
            h_c_n = tf.reduce_sum(tf.einsum('ijk,kl->ijl', h_tmp, self._W), axis=1)

            query_c = tf.expand_dims(self._query_layer_1(h_c_n), axis=1)

            u_c_n = self.conf.c * tf.nn.tanh(
                tf.reduce_sum(tf.multiply(query_c, key_1), axis=2) / tf.convert_to_tensor(
                    np.sqrt(self.conf.embedding_dim).astype(np.float32)))

            if t != 0:
                indices = tf.cast(tf.where(tf.logical_not(mask)), tf.int64)
                val_masked = tf.SparseTensor(indices, tf.broadcast_to(tf.float32.min, [self.conf.batch * t]),
                                             (self.conf.batch, self.conf.n_node))

                u_c_n = tf.add(u_c_n, tf.sparse.to_dense(val_masked))

            output_log_proba.append(tf.nn.log_softmax(u_c_n, axis=1))

            pi_t = tf.argmax(output_log_proba[-1], axis=1)

            pi = tf.scatter_nd_update(pi,
                                      tf.stack([tf.range(self.conf.batch, dtype=tf.int64),
                                                tf.cast(tf.broadcast_to(t, [self.conf.batch]), tf.int64)], axis=1),
                                      pi_t)

        stacked_log_proba = tf.stack(output_log_proba, axis=1)
        index = tf.stack([tf.convert_to_tensor(
            np.array([i for i in range(self.conf.batch) for _ in range(self.conf.n_node)], dtype=np.int64)),
                          tf.convert_to_tensor(
                              np.array([i for _ in range(self.conf.batch) for i in range(self.conf.n_node)],
                                       dtype=np.int64)),
                          tf.reshape(pi, (self.conf.n_node * self.conf.batch,))],
                         axis=-1)

        computed_log_likelihood = tf.reshape(tf.gather_nd(stacked_log_proba, index),
                                             shape=(self.conf.batch, self.conf.n_node))

        computed_log_likelihood = tf.reduce_sum(computed_log_likelihood, axis=1)

        return pi, computed_log_likelihood
