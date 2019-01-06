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
import utils
from encoder import Encoder
from decoder import Decoder


class AttentionTspModel(snt.AbstractModule):

    def __init__(self, conf, name="attention_tsp_model"):
        super(AttentionTspModel, self).__init__(name=name)
        self.training = True
        self.conf = conf
        with self._enter_variable_scope():
            self._encoder = Encoder(conf)
            self._decoder = Decoder(conf)

    def modify_state(self, training=True, baseline=False):
        self.training = training
        self._encoder.modify_state(training)
        self._decoder.modify_state(training, baseline)

    def load(self, dic, sess):
        with self._enter_variable_scope():
            self._encoder.load(dic["encoder"], sess)
            self._decoder.load(dic["decoder"], sess)

    def save(self, sess):
        with self._enter_variable_scope():
            dic = {
                "encoder": self._encoder.save(sess),
                "decoder": self._decoder.save(sess)
            }
        return dic

    def _build(self, graph):
        _log_p, pi = self._decoder(self._encoder(graph))
        ll = self._calc_log_likelihood(_log_p, pi)
        r_graph, cost = self.get_cost(graph, pi)
        return ll, cost

    def _calc_log_likelihood(self, _log_p, pi):
        p = tf.nn.log_softmax(_log_p, -1)
        ll = tf.batch_gather(p, tf.expand_dims(pi, -1))
        return tf.squeeze(tf.reduce_sum(ll, 1))

    def get_cost(self, graph, pi):
        nodes = tf.reshape(graph.nodes, (self.conf.batch, self.conf.n_node, self.conf.init_dim))
        ordered_nodes = tf.batch_gather(nodes, pi)  # (b,s,2)
        new_cost = tf.reduce_sum(tf.norm(ordered_nodes - tf.roll(ordered_nodes, -1, 1), axis=-1), -1)
        return graph, new_cost

