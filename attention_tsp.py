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

    def _build(self, graph):
        """Perform a Multi Head Attention over a graph

        Args:
            graph (graphs.GraphsTuple): The graph over which the multi head attention will be performed.

        Returns:
            graphs.GraphsTuple

        """
        assert tf.reduce_all(tf.equal(graph.n_node[0], graph.n_node)), "Not all the graphs have the same size!"

        nodes = graph.nodes

        query = self._query_layer(nodes).reshape(-1, HEAD_NBR, QUERY_DIM)
        value = self._value_layer(nodes).reshape(-1, HEAD_NBR, VALUE_DIM)
        key = self._key_layer(nodes).reshape(-1, HEAD_NBR, KEY_DIM)

        attention_result = self._graph_mha(value, key, query, graph)
        # TODO: multiply each node my a matrix and perform a reduce sum
        return attention_result


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
            # TODO: Cannot put mha  in sequential because deals with a graph while the sequential deal with nodes values
            self._full_encoder = snt.Sequential([self._mha,
                                                 self._batch_norm,
                                                 self._feed_forward_residual,
                                                 self._batch_norm],
                                                name="full_encoder")
            self._full_encoder_block = blocks.NodeBlock(self._full_encoder,
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
        return self._full_encoder_block(graph)


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
            self._initial_projection_block = blocks.NodeBlock(self._initial_projection,
                                                              use_received_edges=False,
                                                              use_nodes=True,
                                                              use_globals=False,
                                                              name="initial_block_projection")
            self._encoder_layers = snt.Sequential([EncoderLayer("encoder_layer_%i" % i) for i in range(head_nbr)],
                                                  name="transformer_group")

    def _build(self, graph):
        """Encodes the graph

        Args:
            graph (graphs.GraphsTuple): A `graphs.GraphsTuple` ...

        Returns:
            An ouput `graphs.GraphsTuple` with encoded nodes.
        """
        projected_graph = self._initial_projection_block(graph)
        return self._encoder_layers(projected_graph)
