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


class MultiHeadAttentionResidual(snt.AbstractModule):
    """MultiHeadAttention layer

    This modules computes the values, key and query for a `graphs.GraphsTuples` nodes and then calls
    `modules.selfAttention`
    """

    def __init__(self, name="multi_head_attention"):
        """ Inits the module.

               Args:
                   name: The module name.
               """
        super(MultiHeadAttentionResidual, self).__init__(name=name)
        self.training = True
        with self._enter_variable_scope():
            initializers = {"w": tf.initializers.random_uniform(-INIT_DIM, INIT_DIM)}
            self._query_layer = snt.Linear(output_size=HEAD_NBR * QUERY_DIM,
                                           use_bias=False,
                                           initializers=initializers,
                                           name="query_computer")
            self._value_layer = snt.Linear(output_size=HEAD_NBR * VALUE_DIM,
                                           use_bias=False,
                                           initializers=initializers,
                                           name="value_computer")
            self._key_layer = snt.Linear(output_size=HEAD_NBR * KEY_DIM,
                                         use_bias=False,
                                         initializers=initializers,
                                         name="key_computer")
            self._graph_mha = modules.SelfAttention("graph_self_attention")
            self._W = tf.get_variable(name="multi_head_reducer",
                                      shape=(VALUE_DIM, EMBEDDING_DIM),
                                      initializer=tf.initializers.random_uniform(-INIT_DIM, INIT_DIM))

    def _build(self, graph):
        """Perform a Multi Head Attention over a graph

        Args:
            graph (graphs.GraphsTuple): The graph over which the multi head attention will be performed.

        Returns:
            graphs.GraphsTuple

        """
        # assert tf.reduce_all(tf.equal(graph.n_node[0], graph.n_node)), "Not all the graphs have the same size!"

        nodes = graph.nodes

        query = tf.reshape(self._query_layer(nodes), (-1, HEAD_NBR, QUERY_DIM))
        value = tf.reshape(self._value_layer(nodes), (-1, HEAD_NBR, VALUE_DIM))
        key = tf.reshape(self._key_layer(nodes), (-1, HEAD_NBR, KEY_DIM))

        attention_result = self._graph_mha(value, key, query, graph)
        nodes_a = attention_result.nodes
        new_nodes = tf.einsum('ijk,kl->ijl', nodes_a, self._W)
        new_nodes = tf.reduce_sum(new_nodes, axis=1)
        return attention_result.replace(nodes=tf.add(new_nodes, nodes))


class EncoderLayer(snt.AbstractModule):
    """Layer for the Encoding module.

    This layer contains:

    - Multi-head attention
    - Fully connected Feed-Forward network
    Each layer adds a skip-connection and a batch normalization
    """

    def __init__(self, name="encoder_layer"):
        """ Inits the module.

        Args:
            name: The module name.
        """
        super(EncoderLayer, self).__init__(name=name)
        self.training = True
        with self._enter_variable_scope():
            initializers = {"w": tf.initializers.random_uniform(-INIT_DIM, INIT_DIM),
                            "b": tf.initializers.random_uniform(-INIT_DIM, INIT_DIM)}

            self._mha = MultiHeadAttentionResidual()

            self._batch_norm = snt.BatchNorm(scale=True)

            self._lin_to_hidden = snt.Linear(output_size=FF_HIDDEN_SIZE,
                                             initializers=initializers,
                                             name="lin_to_hidden")
            self._hidden_to_ouput = snt.Linear(output_size=EMBEDDING_DIM,
                                               initializers=initializers,
                                               name="hidden_to_ouput")
            self._feed_forward = snt.Sequential([self._lin_to_hidden,
                                                 tf.nn.relu,
                                                 self._hidden_to_ouput],
                                                name="feed_forward")
            self._feed_forward_residual = snt.Residual(self._feed_forward, name="feed_forward_residual")

            # Todo: Check if same batch norm
            self._part_encoder = snt.Sequential([lambda x: self._batch_norm(x, is_training=self.training),
                                                 self._feed_forward_residual,
                                                 lambda x: self._batch_norm(x, is_training=self.training)],
                                                name="full_encoder")
            self._part_encoder_block = blocks.NodeBlock(lambda: self._part_encoder,
                                                        use_received_edges=False,
                                                        use_nodes=True,
                                                        use_globals=False,
                                                        name="encoder_block")

    def _build(self, graph):
        """

        Args:
            graph (graphs.GraphsTuple):

        Returns:
            graphs.GraphsTuple

        """
        mha_residual_graph = self._mha(graph)
        new_graph = self._part_encoder_block(mha_residual_graph)

        return new_graph


class Encoder(snt.AbstractModule):
    """Encoder for attention-tsp module.

    This modules projects the coordinates into a INIT_EMBEDDING_DIM dimension space
    and then stacks ENCODER_NBR_LAYERS layers of EncoderLayer.
    """

    def __init__(self, name="encoder-attention-tsp", head_nbr=HEAD_NBR):
        """Inits the module.

        Args:
            name: The module name.
        """

        super(Encoder, self).__init__(name=name)
        self.head_nbr = head_nbr
        self.training = True
        with self._enter_variable_scope():
            initializers = {"w": tf.initializers.random_uniform(-INIT_DIM, INIT_DIM),
                            "b": tf.initializers.random_uniform(-INIT_DIM, INIT_DIM)}
            self._initial_projection = snt.Linear(output_size=EMBEDDING_DIM, initializers=initializers,
                                                  name="initial_projection")
            self._initial_projection_block = blocks.NodeBlock(lambda: self._initial_projection,
                                                              use_received_edges=False,
                                                              use_nodes=True,
                                                              use_globals=False,
                                                              name="initial_block_projection")
            self._encoder_layers = [EncoderLayer("encoder_layer_%i" % i) for i in range(ENCODER_NBR_LAYERS)]

    def _build(self, graph):
        """Encodes the graph

        Args:
            graph (graphs.GraphsTuple): A `graphs.GraphsTuple` ...

        Returns:
            An ouput `graphs.GraphsTuple` with encoded nodes.
        """
        # assert tf.reduce_all(tf.equal(graph.n_node[0], graph.n_node)), "Not all the graphs have the same size!"

        projected_graph = self._initial_projection_block(graph)
        for encoder in self._encoder_layers:
            projected_graph = encoder(projected_graph)
        # could use block func
        nodes = tf.reshape(projected_graph.nodes, (BATCH, N_NODE, EMBEDDING_DIM))

        return projected_graph.replace(globals=tf.reduce_mean(nodes, axis=1))
