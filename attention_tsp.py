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


class TSP_Model:

    def __init__(self, opt):
        self.lr = opt.learning_rate
        self.save_dir = opt.save_dir
        self.cuda = opt.gpu != -1
        self.gpu = opt.gpu
        self.encoder = encoder.Encoder()
        self.decoder = decoder.Decoder()
        self.global_cost = blocks.EdgesToGlobalsAggregator(tf.unsorted_segment_sum)

    def set_input(self, input_placeholder):
        self.input_placeholder = input_placeholder

    def forward(self, graph):
        pi, log_p = self.decoder(self.encoder(graph))
        result_graph = self.create_result_graph(graph, pi)
        return result_graph, pi, log_p

    def create_result_graph(self, graph, pi):
        receivers = tf.roll(pi, shift=-1, axis=1)
        receivers = tf.reshape(receivers, (BATCH * N_NODE, 1))
        senders = tf.reshape(pi, (BATCH * N_NODE, 1))
        edges = tf.zeros(BATCH * N_NODE)
        n_edge = tf.convert_to_tensor([N_NODE] * BATCH)
        graph = graph.replace(receivers=receivers, senders=senders, edges=edges, n_edge=n_edge)
        distance = tf.sqrt(tf.reduce_sum(tf.squared_difference(blocks.broadcast_receiver_nodes_to_edges(graph),
                                                               blocks.broadcast_sender_nodes_to_edges(graph)), axis=1))
        graph = graph.replace(edges=distance)
        cost = self.global_cost(graph)
        graph = graph.replace(globals=cost)
        return graph

    def compute_loss(self, graph, baseline):
        result_graph, pi, log_p = self.forward(graph)
        loss = tf.reduce_mean(result_graph.globals)
        grad_loss = result_graph.globals
        if baseline is not None:
            grad_loss = tf.subtract(grad_loss, baseline)
        grad_loss = tf.multiply(grad_loss, log_p)
        grad_loss = tf.reduce_sum(grad_loss)
        return loss, grad_loss

    def freeze(self, require_grad=False):
        pass
