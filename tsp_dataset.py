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
    coordinates = np.random.random((num_nodes, TSP_DIMENSION))
    graph = nx.OrderedMultiGraph()
    graph.graph['features'] = np.array(0.)
    graph.add_nodes_from(zip(*(range(num_nodes), map(to_feature_dic, coordinates))))
    # i_, j_ = np.meshgrid(range(num_nodes), range(num_nodes), indexing="ij")
    # weighted_edges = list(zip(i_.ravel(), j_.ravel(), [to_feature_dic(np.array(0.))]*(num_nodes**2)))
    # graph.add_edges_from(weighted_edges)
    return graph


def generate_batch(num_graph=BATCH_SIZE, num_nodes=NODES_NUMBER):
    return [generate_graph(num_nodes) for _ in range(num_graph)]
