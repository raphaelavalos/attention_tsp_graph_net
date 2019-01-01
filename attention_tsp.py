from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import encoder
import decoder

import tensorflow as tf
from graph_nets import utils_tf
from graph_nets import utils_np
from graph_nets import graphs
from graph_nets import blocks
from graph_nets import modules
import sonnet as snt


class TSP_Model:

    def __init__(self, opt, conf):
        self.opt = opt
        self.conf = conf
        self.encoder = encoder.Encoder(conf)
        self.decoder = decoder.Decoder(conf)

    def forward(self, graph):
        pi, computed_log_likelihood = self.decoder(self.encoder(graph))
        result_graph, cost = self.create_result_graph(graph, pi)
        return result_graph, pi, computed_log_likelihood, cost

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

    def compute_loss(self, graph):
        result_graph, pi, computed_log_likelihood, cost = self.forward(graph)
        loss = tf.reduce_mean(tf.multiply(cost, computed_log_likelihood))
        # if baseline is not None:
        #     grad_loss = tf.subtract(grad_loss, baseline)
        # print('grad_loss', grad_loss)
        # print('log_p', log_p)
        # print('grad_loss', grad_loss)
        return result_graph, loss, computed_log_likelihood, pi

    def freeze(self, require_grad=False):
        pass
