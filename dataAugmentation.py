import re
from networkx.classes.function import degree
import numpy as np
import scipy.sparse as sp
import networkx as nx
from networkx.generators.ego import *
from config import log
np.seterr(divide='ignore', invalid='ignore')

degrees_logged_cache = {}

level_n_nodes_and_edges_cache = {}

ego_graph_cache = {}


def get_ego_graph(graph, target_node, level):
    global ego_graph_cache

    graphHash = hash(graph)
    key = f"{graphHash}_{target_node}_{level}"

    if (key not in ego_graph_cache):
        ego_graph_cache[key] = ego_graph(graph, target_node, level, True)

    return ego_graph_cache[key]


def extract_level_n_nodes_and_edges(graph, target_node, level):
    """Given a graph and the target node, return the nodes and edges from that level"""
    global level_n_nodes_and_edges_cache

    if (level < 1):
        raise Exception("Level cannot be smaller than 1")

    graphHash = hash(graph)
    key = f"{graphHash}_{target_node}_{level}"

    if (key not in level_n_nodes_and_edges_cache):
        nth_hop_neighbor = get_ego_graph(graph, target_node, level)
        n_1th_hop_neighbor = get_ego_graph(graph, target_node, level-1)
        nodes = np.array(list(nth_hop_neighbor.nodes() -
                              n_1th_hop_neighbor.nodes()))
        edges = np.array(list(nth_hop_neighbor.edges() -
                              n_1th_hop_neighbor.edges()))

        level_n_nodes_and_edges_cache[key] = (nodes, edges)

    return level_n_nodes_and_edges_cache[key]


def get_affected_nodes(graph, target_node, node_selection_strategy):
    affected_nodes = []

    if (node_selection_strategy == "random"):
        affected_nodes, _ = extract_level_n_nodes_and_edges(
            graph, target_node, 3)

    if (node_selection_strategy == "one_hop_neighbors"):
        affected_nodes, _ = extract_level_n_nodes_and_edges(
            graph, target_node, 1)

    return affected_nodes


def get_degrees_logged(graph):
    """Get the list of degrees in Graph"""
    global degrees_logged_cache

    graph_hash = hash(graph)

    if (graph_hash not in degrees_logged_cache):
        # https://stackoverflow.com/questions/47433523/get-degree-of-each-nodes-in-a-graph-by-networkx-in-python
        degrees = np.array([val for (_, val) in sorted(
            graph.degree(), key=lambda pair: pair[0])])

        degrees_logged = np.nan_to_num(np.log(degrees), True, 0, 0, 0)
        # since we are using degree
        degrees_logged_cache[graph_hash] = degrees_logged

    return degrees_logged_cache[graph_hash]


def calculate_new_edges(nodes, degrees_logged, p, level, add_edge_probability, node_selection_strategy):

    if (len(nodes) == 0):
        #log(f"\tADD EDGES: On Level {level}, there were no nodes")
        return []

    # Get the scores for the nodes on this level. s_n = log(d)
    node_scores = degrees_logged[nodes]

    # Get min, average and minimum scores on this level.
    min_node_score = np.min(node_scores)
    avg_node_score = np.average(node_scores)

    # Calculate the add probability
    # P_(e-add) = min [ (p/l) (s_n - s_n-min) / (s_n-avg - s_n-min), 1 ]
    add_probability = (p / level) * (node_scores -
                                     min_node_score) / (avg_node_score - min_node_score)
    add_probability = np.nan_to_num(add_probability, True, 0, 0, 0)
    add_probability[add_probability > 1] = 1

    # Get edges to be added (Probability > add_edge_probability)
    edge_addition_indices = np.where(add_probability > add_edge_probability)

    edges_to_be_added = nodes[edge_addition_indices]

    #log(f"\tADD EDGES: On Level {level}, adding {len(edges_to_be_added)} new edges out of {len(nodes)} potential edges.")

    return edges_to_be_added


def calculate_removed_edges(nodes, degrees_logged, edges, p, level, remove_edge_probability, node_selection_strategy):

    if (len(edges) == 0 or len(nodes) == 0):
        #log(f"\tREMOVE EDGES: On Level {level}, there were no edges")
        return []

    # Get the node on the lower level. If an edge is on the same level, the first node is used
    # Get a boolean 2D array with T / F if the node is in the edge
    edges_reduced = np.isin(edges, nodes)

    # If an edge's nodes are in the list, then we only use the first one.
    edges_reduced[:, 1] = np.logical_and(
        edges_reduced[:, 1], np.logical_xor(edges_reduced[:, 0], edges_reduced[:, 1]))

    # Get the "lowest node"
    lowest_nodes = edges[edges_reduced]

    # Retrive the logged degrees as the edge scores
    edges_score = degrees_logged[lowest_nodes]

    # Calculate max and average edge score
    max_edge_score = np.max(edges_score)
    average_edge_score = np.average(edges_score)

    # Calculate the remove probability

    remove_probabilty = p * level * \
        (max_edge_score - edges_score) / (max_edge_score - average_edge_score)
    # Replace value to 1 if greater than 1 since it is a probability
    remove_probabilty = np.nan_to_num(remove_probabilty, True, 0, 0, 0)
    remove_probabilty[remove_probabilty > 1] = 1

    # Get edges to be removed (the probability to remove is greater than the set threshold)
    edge_removal_indices = np.where(
        remove_probabilty > remove_edge_probability)

    edges_to_be_removed = edges[edge_removal_indices]

    #log(f"\tREMOVE EDGES: On Level {level}, removing {len(edges_to_be_removed)} out of {len(edges)}")

    return edges_to_be_removed


def add_edges(adjacency_matrix, target_node, p, add_edge_probability, levels, node_selection_strategy):
    """Given an adjacency matrix with the target node and hyperparameter p, we calculate the probability of adding edges on level 2 and level 3. """
    graph = nx.Graph(adjacency_matrix)

    #log(f"ADD EDGES: Original graph has {len(graph.edges)} edges.")

    degrees_logged = get_degrees_logged(graph)

    affected_nodes = get_affected_nodes(
        graph, target_node, node_selection_strategy)

    for level in levels:
        nodes, _ = extract_level_n_nodes_and_edges(graph, target_node, level)

        edges_to_be_added = calculate_new_edges(
            nodes, degrees_logged, p, level, add_edge_probability, node_selection_strategy)

        for new_node in edges_to_be_added:
            graph.add_edge(target_node, new_node)

            if (node_selection_strategy == "directly_affected_nodes"):
                affected_nodes.append(new_node)

    #log(f"ADD EDGES: New graph has {len(graph.edges)} edges.")

    return nx.adjacency_matrix(graph), affected_nodes


def remove_edges(adjacency_matrix, target_node, p, remove_edge_probability, levels, node_selection_strategy):
    """Given an adjacency matrix with the target node and the hyperparameter p, we calculate the probability of removing edges on level 2 and level 3"""
    graph = nx.Graph(adjacency_matrix)

    #log(f"REMOVE EDGES: Original graph has {len(graph.edges)} edges.")

    degrees_logged = get_degrees_logged(graph)

    affected_nodes = get_affected_nodes(
        graph, target_node, node_selection_strategy)

    for level in levels:
        nodes, edges = extract_level_n_nodes_and_edges(
            graph, target_node, level)

        edges_to_be_removed = calculate_removed_edges(nodes, degrees_logged,
                                                      edges, p, level, remove_edge_probability, node_selection_strategy)

        for edge in edges_to_be_removed:
            graph.remove_edge(edge[0], edge[1])

            if (node_selection_strategy == "directly_affected_nodes"):
                affected_nodes.append(edge[0])
                affected_nodes.append(edge[1])

    #log(f"REMOVE EDGES: New graph has {len(graph.edges)} edges.")

    return nx.adjacency_matrix(graph), affected_nodes
