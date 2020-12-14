import argparse
import distutils.util

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def convert_to_boolean(original_value, default_value):
    try:
        return distutils.util.strtobool(original_value)
    except Exception:
        return default_value


def convert_to_int(original_value, default_value):
    try:
        return int(original_value)
    except Exception:
        return default_value


args = argparse.ArgumentParser()
args.add_argument('--dataset', default='cora')
args.add_argument('--model', default='gcn')
args.add_argument('--learning_rate', default=0.01)
args.add_argument('--epochs', default=400)
args.add_argument('--hidden1', default=16)
args.add_argument('--dropout', default=0.5)
args.add_argument('--weight_decay', default=5e-4)
args.add_argument('--early_stopping', default=10)
args.add_argument('--max_degree', default=3)
args.add_argument('--debug', default=True)

# Argument Values for NodeAug
args.add_argument('--nodeaug', default=False)    # Set to true to use NodeAug


# Hyperparameter in L = L_s + alpha * L_C
args.add_argument('--alpha', default=1)

# Hyperparameter p in all the probability calculations
args.add_argument('--p', default=0.3)

# Data Augmentation parameters for Adding Edges
args.add_argument('--add_edges_probability', default=0.1)
args.add_argument('--addEdgeLevels', default=[2, 3])

# Data Augmentation parameters for Removing Edges
args.add_argument('--remove_edges_probability', default=0.8)
args.add_argument('--removeEdgeLevels', default=[2, 3])

# K-NodeAug
args.add_argument('--enable_K_NodeAug', default=True)

# K-NodeAug: Strategy: 'random', 'directly_affected_nodes', 'one_hop_neighbors'
args.add_argument('--node_selection_strategy', default="random")
args.add_argument('--k', default=20)


# Output File
args.add_argument('--file_path', default="")

args = args.parse_args()

args.debug = convert_to_boolean(args.debug, True)
args.nodeaug = convert_to_boolean(args.nodeaug, False)
args.enable_K_NodeAug = convert_to_boolean(args.enable_K_NodeAug, True)
args.k = convert_to_int(args.k, 20)
args.epochs = convert_to_int(args.epochs, 400)


def log(value):
    if (args.debug):
        print(value)


log(args)
