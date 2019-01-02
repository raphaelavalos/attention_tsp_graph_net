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
        pi, computed_log_likelihood = self._decoder(self._encoder(graph))
        result_graph, cost = self.create_result_graph(graph, pi)
        return pi, computed_log_likelihood, result_graph, cost

    def create_result_graph(self, graph, pi):
        receivers = tf.roll(pi, shift=-1, axis=1)
        receivers = tf.reshape(receivers, (self.conf.batch * self.conf.n_node, 1))
        senders = tf.reshape(pi, (self.conf.batch * self.conf.n_node, 1))
        edges = tf.zeros(self.conf.batch * self.conf.n_node)
        n_edge = tf.convert_to_tensor([self.conf.n_node] * self.conf.batch)
        graph = graph.replace(receivers=receivers, senders=senders, edges=edges, n_edge=n_edge)
        distance = tf.sqrt(tf.reduce_sum(
            tf.squared_difference(
                tf.reshape(blocks.broadcast_receiver_nodes_to_edges(graph),
                           (self.conf.batch * self.conf.n_node, self.conf.init_dim)),
                tf.reshape(blocks.broadcast_sender_nodes_to_edges(graph),
                           (self.conf.batch * self.conf.n_node, self.conf.init_dim))), axis=1))
        cost = tf.reduce_sum(tf.reshape(distance, (self.conf.batch, self.conf.n_node)), axis=1)
        graph = graph.replace(edges=distance, globals=cost)
        return graph, cost

