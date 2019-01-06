from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import tensorflow as tf
from graph_nets import utils_tf
from graph_nets import utils_np
from graph_nets import graphs
from graph_nets import blocks
from graph_nets import modules
import sonnet as snt
import utils


class Decoder(snt.AbstractModule):

    def __init__(self, conf, name="decoder-attention-tsp"):
        super(Decoder, self).__init__(name=name)
        self.conf = conf
        self.training = True
        self.baseline = False
        with self._enter_variable_scope():
            base_init = tf.initializers.random_uniform(-self.conf.init_max, self.conf.init_max)

            self.W_placeholder = tf.get_variable(name="w_placeholder",
                                                 shape=(1, 2 * conf.embedding_dim),
                                                 initializer=utils.initializer(conf.embedding_dim),
                                                 dtype=tf.float32)

            self._precompute_l = snt.Linear(name="precompute_linear",
                                            output_size=3 * self.conf.embedding_dim,
                                            use_bias=False,
                                            initializers={'w': utils.initializer(conf.embedding_dim)})

            self._fixed_context_l = snt.Linear(name="fixed_context_linear",
                                               output_size=self.conf.embedding_dim,
                                               use_bias=False,
                                               initializers={'w': utils.initializer(conf.embedding_dim)})

            self._step_context_l = snt.Linear(name="step_context_linear",
                                              output_size=self.conf.embedding_dim,
                                              use_bias=False,
                                              initializers={'w': utils.initializer(2 * conf.embedding_dim)})

            self._glimpse_l = snt.Linear(name='glimpse_linear',
                                         output_size=self.conf.embedding_dim,
                                         use_bias=False,
                                         initializers={'w': utils.initializer(conf.embedding_dim)})

    def _precompute(self, nodes, graph_em):
        # nodes shape should be [batch * n_node, e_dim]
        with tf.name_scope('precompute'):
            out = tf.reshape(self._precompute_l(nodes),
                             (self.conf.batch, self.conf.n_node, 3 * self.conf.embedding_dim))
            key_context, value_context, key = tf.split(out, num_or_size_splits=3, axis=-1)
            key_context = tf.reshape(key_context,
                                     (self.conf.batch, self.conf.n_node, self.conf.head_nbr, self.conf.key_dim))
            value_context = tf.reshape(value_context,
                                       (self.conf.batch, self.conf.n_node, self.conf.head_nbr, self.conf.value_dim))
            fixed_context = tf.reshape(self._fixed_context_l(graph_em),
                                       (self.conf.batch, self.conf.head_nbr, self.conf.query_dim))

        return key_context, value_context, key, fixed_context

    def _get_log_p(self, nodes, attention_node_fixed, state):
        with tf.name_scope('get_log_p'):
            key_context, value_context, key, fixed_context = attention_node_fixed
            prev_a, first_a, mask = state
            if first_a is None:
                step_context = self._step_context_l(self.W_placeholder)
                step_context = tf.reshape(step_context, (1, self.conf.head_nbr, self.conf.query_dim))
                step_context = tf.broadcast_to(step_context, (self.conf.batch, self.conf.head_nbr, self.conf.key_dim))
            else:
                step_context = tf.concat([tf.batch_gather(nodes, tf.expand_dims(prev_a, -1)),
                                          tf.batch_gather(nodes, tf.expand_dims(first_a, -1))], axis=-1)
                step_context = tf.squeeze(step_context, axis=1)
                step_context = self._step_context_l(step_context)
                step_context = tf.reshape(step_context, (self.conf.batch, self.conf.head_nbr, self.conf.key_dim))
            query_context = fixed_context + step_context
            logits = self._one_to_many_logits(query_context, key_context, value_context, key, mask)
        return logits

    def _one_to_many_logits(self, query_context, key_context, value_context, key, mask):
        with tf.name_scope('one_to_many_logits'):
            compatibility = tf.expand_dims(query_context, 1) * key_context
            compatibility = tf.reduce_sum(compatibility, axis=-1) / np.sqrt(self.conf.key_dim, dtype=np.float32)
            compatibility -= 100000 * tf.expand_dims(mask, axis=-1)
            attention_weight = tf.nn.softmax(compatibility, axis=1, name="attention_weight")
            glimpse_multi = tf.multiply(tf.expand_dims(attention_weight, -1), value_context)
            glimpse_multi = tf.reduce_sum(glimpse_multi, axis=1)
            glimpse = self._glimpse_l(tf.reshape(glimpse_multi, (self.conf.batch, self.conf.embedding_dim)))
            logits0 = tf.reduce_sum(tf.expand_dims(glimpse, 1) * key, axis=-1) / np.sqrt(self.conf.embedding_dim,
                                                                                         dtype=np.float32)
            logits = self.conf.c * tf.tanh(logits0, name="clip_logits")
            logits -= 100000 * mask
            return logits

    def _select_node(self, logits):
        with tf.name_scope('select_node'):
            if self.baseline:
                selected = tf.argmax(logits, axis=1, output_type=tf.int32)
            else:
                selected = tf.reshape(tf.random.multinomial(logits, num_samples=1, output_dtype=tf.int32),
                                      shape=(self.conf.batch,))
        return selected

    def _update_state(self, state, selected):
        prev_a, first_a, mask = state
        prev_a = selected
        if first_a is None:
            first_a = prev_a
        mask = mask + tf.one_hot(selected, depth=self.conf.n_node)
        return prev_a, first_a, mask

    def _init_state(self):
        mask = tf.zeros(shape=(self.conf.batch, self.conf.n_node))
        return None, None, mask

    def _build(self, graph):
        """Decodes the graph

        Args:
            graph (graphs.GraphsTuple):

        Returns:
            The ordered indices
        """

        outputs = []
        sequences = []
        nodes = graph.nodes
        graph_em = graph.globals
        attention_node_fixed = self._precompute(nodes, graph_em)

        nodes = tf.reshape(nodes, (self.conf.batch, self.conf.n_node, self.conf.embedding_dim))

        state = self._init_state()

        for t in range(self.conf.n_node):
            logits = self._get_log_p(nodes, attention_node_fixed, state)
            selected = self._select_node(logits)
            state = self._update_state(state, selected)
            outputs.append(logits)
            sequences.append(selected)

        return tf.stack(outputs, axis=1), tf.stack(sequences, axis=1)

    def modify_state(self, training=True, baseline=False):
        self.training = training
        self.baseline = baseline

    def load(self, dic, sess):
        with self._enter_variable_scope():
            self.W_placeholder.load(dic["W_placeholder"], sess)
            utils.load_linear(self._precompute_l, dic["precompute_linear"], sess)
            utils.load_linear(self._fixed_context_l, dic["fixed_context_linear"], sess)
            utils.load_linear(self._step_context_l, dic["step_context_linear"], sess)
            utils.load_linear(self._glimpse_l, dic["glimpse_linear"], sess)

    def save(self, sess):
        with self._enter_variable_scope():
            dic = {
                "W_placeholder": self.W_placeholder.eval(sess),
                "precompute_linear": utils.save_linear(self._precompute_l, sess),
                "fixed_context_linear": utils.save_linear(self._fixed_context_l, sess),
                "step_context_linear": utils.save_linear(self._step_context_l, sess),
                "glimpse_linear": utils.save_linear(self._glimpse_l, sess),
            }
        return dic
