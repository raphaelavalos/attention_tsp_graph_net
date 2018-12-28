import numpy as np
import networkx as nx
from scipy import spatial

TSP_DIMENSION = 2
DISTANCE_WEIGHT_NAME = 'features'
BATCH_SIZE = 64
NODES_NUMBER = 20


def to_feature_dic(x):
    return dict(features=x)


def generate_graph(num_nodes=20):
    """
    Generate a networkx graph for the TSP problem
    Args:
        num_nodes (int): the number of nodes

    Returns:
        OrderedMultiGraph
    """
    coordinates = np.random.random((num_nodes, TSP_DIMENSION)).astype(np.float32)
    graph = nx.OrderedDiGraph()
    graph.graph['features'] = np.array(0.)
    graph.add_nodes_from(zip(*(range(num_nodes), map(to_feature_dic, coordinates))))
    edges = np.dstack(np.meshgrid(range(num_nodes), range(num_nodes))).reshape(-1, TSP_DIMENSION)
    graph.add_edges_from(edges, features=np.array(0.))
    return graph


def generate_networkx_batch(num_graph=BATCH_SIZE, num_nodes=NODES_NUMBER):
    """
    Generate a batch of networkx graph data for the TSP problem
    Args:
        num_graph (int): the size of the batch
        num_nodes (int): the number of nodes

    Returns:
        list of OrderedMultiGraph

    """
    return [generate_graph(num_nodes) for _ in range(num_graph)]
