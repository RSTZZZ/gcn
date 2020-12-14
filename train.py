from networkx.linalg.graphmatrix import adjacency_matrix
from config import args
from config import log

import tensorflow as tf
import time
from tensorflow.keras import optimizers
from utils import *
from models import GCN
from dataAugmentation import *
from scipy.stats import entropy


log(f"Tensorflow Version: {tf.__version__}")
assert tf.__version__.startswith('2.')

data_augmentation_cache = {}

add_edge_adj_cache = {}
remove_edge_adj_cache = {}


def get_add_edges(adjacency_matrix, target_node):
    if (target_node not in add_edge_adj_cache):
        data_augmented_matrix, affected_nodes = add_edges(
            adjacency_matrix, target_node, args.p, args.add_edges_probability, args.addEdgeLevels, args.node_selection_strategy)

        add_edge_adj_cache[target_node] = data_augmented_matrix, affected_nodes

    return add_edge_adj_cache[target_node]


def get_remove_edges(adjacency_matrix, target_node):
    if (target_node not in remove_edge_adj_cache):
        data_augmentented_matrix, affected_nodes = remove_edges(adjacency_matrix, target_node, args.p,
                                                                args.remove_edges_probability, args.removeEdgeLevels, args.node_selection_strategy)

        data_augmented_support = preprocess_support(
            [preprocess_adj(data_augmentented_matrix)])

        remove_edge_adj_cache[target_node] = data_augmented_support, affected_nodes

    return remove_edge_adj_cache[target_node]


def perform_data_augmentation(adjacency_matrix, target_node):

    if (target_node not in data_augmentation_cache):
        # Data Augmentation

        # 1. Replace Attributes - Update features by replacing attribute values in the features

        # 2. Add Edges - Update support using adjacency matrix and adding edges on adjacency matrix
        edges_added_adjacency_matrix, edges_added_affected_nodes = get_add_edges(
            adjacency_matrix, target_node)

        # 3. Remove Edges - Update support using adjacency matrix and removing edges on adjacency matrix
        edges_removed_adjacency_matrix, edges_removed_affected_nodes = get_remove_edges(
            edges_added_adjacency_matrix, target_node)

        # Merge the two sets of affacted nodes.
        affected_nodes = set(edges_added_affected_nodes) | set(
            edges_removed_affected_nodes)

        affected_nodes = list(affected_nodes)

        data_augmentation_cache[target_node] = edges_removed_adjacency_matrix, affected_nodes

    return data_augmentation_cache[target_node]


def get_k_1_nodes(affected_nodes, unrun_nodes):
    k_1_nodes = []

    if (args.enable_K_NodeAug and args.k > 1):

        # Randomize the list of affected nodes
        random_nodes = np.random.choice(
            affected_nodes, len(affected_nodes), replace=False)

        # Choose the first k - 1 nodes that have not been "run"
        for node in random_nodes:
            if node in unrun_nodes:
                k_1_nodes.append(node)

            if (len(k_1_nodes) == args.k - 1):
                break

    return k_1_nodes


# set random seed
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)


# load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(
    args.dataset)


log(f"Adjacency List Shape: {adj.shape}")
log(f"Features shape: {features.shape}")
log(
    f"Y (train, validation, test) shape: ({y_train.shape}, {y_val.shape}, {y_test.shape})")
log(
    f"Mask (train, validation, test) shape: ({train_mask.shape}, {val_mask.shape}, {test_mask.shape})")


# D^-1@X
features = preprocess_features(features)  # [49216, 2], [49216], [2708, 1433]

log(f"Feature Coordinates: {features[0].shape}")
log(f"Features Data Shape: {features[1].shape}")
log(f"Features Shape: {features[2]}")


if args.model == 'gcn':
    # D^-0.5 A D^-0.5
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
else:
    raise ValueError('Invalid argument for model: ' + str(args.model))


# Create model
model = model_func(input_dim=features[2][1], output_dim=y_train.shape[1],
                   num_features_nonzero=features[1].shape)  # [1433]


train_label = tf.convert_to_tensor(y_train)
train_mask = tf.convert_to_tensor(train_mask)
val_label = tf.convert_to_tensor(y_val)
val_mask = tf.convert_to_tensor(val_mask)
test_label = tf.convert_to_tensor(y_test)
test_mask = tf.convert_to_tensor(test_mask)
features = tf.SparseTensor(*features)
support = preprocess_support(support)
log(f"Support Shape: {support[0].shape}")


node_length = adj.shape[1]

num_features_nonzero = features.values.shape
dropout = args.dropout

optimizer = optimizers.Adam(lr=1e-2)


# Keep track of validation accuracy for early stopping.
validation_history = []


# Keep track of all the parallel universes.
parallel_universe_history = []


start_time = time.time()

# Training
for epoch in range(args.epochs):

    # Training Step
    with tf.GradientTape() as tape:
        supervised_loss, acc, output = model(
            (features, train_label, train_mask, support))

        loss = supervised_loss

        if (args.nodeaug):

            # Create container for all the nodes
            p_u_overall_output = list(range(node_length))

            # Nodes whose output that has not been calculated.
            unrun_nodes = list(range(node_length))

            count = 0

            # Conndition to continue looping
            while (len(unrun_nodes) > 0):

                count += 1

                target_node = np.random.choice(unrun_nodes)

                data_augmented_matrix, affected_nodes = perform_data_augmentation(
                    adj, target_node)

                p_u_loss, p_u_acc, p_u_output = model(
                    (features, train_label, train_mask, data_augmented_matrix))

                p_u_overall_output[target_node] = p_u_output[target_node]
                unrun_nodes.remove(target_node)

                k_1_nodes = get_k_1_nodes(affected_nodes, unrun_nodes)

                for node in k_1_nodes:
                    p_u_overall_output[node] = p_u_output[node]
                    unrun_nodes.remove(node)

            log(f"Ran {count} parallel universes")

            parallel_universe_history.append(count)

            p_u_overall_output = tf.convert_to_tensor(p_u_overall_output)

            # Consistency Loss is 1 / |v | of the Killback-leibler divergence
            loss += (tf.constant(
                [np.sum(entropy(output, p_u_overall_output))]) * args.alpha)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Validation Step
    val_loss, val_acc, _ = model(
        (features, val_label, val_mask, support), training=False)

    validation_history.append(val_loss)

    # Report status for every 20 epoch
    if epoch % 20 == 0:
        log(
            f"{epoch} {float(loss)} {float(acc)} \t {float(val_loss)} {float(val_acc)}")

    # Check if we can stop early
    # Condition 1: The number of epochs we have run is greater than the needed average epochs for early stopping.
    # Condition 2: The last validation loss is greater than the mean of the last ten validation loss.
    if epoch > args.early_stopping and validation_history[-1] > np.mean(validation_history[-(args.early_stopping + 1): -1]):
        log("Early stopping...")
        break


# Find the loss and accuracy on the test set
test_loss, test_acc, _ = model(
    (features, test_label, test_mask, support), training=False)

end_time = time.time()

log(f"TEST_LOSS,TEST_ACCURACY,TOTAL_TIME")

result = f"{float(test_loss)},{float(test_acc)},{float(end_time-start_time)}"

if (args.nodeaug):
    parallel_universe_avg = np.average(parallel_universe_history)
    result += f",{float(parallel_universe_avg)}"


print(result)

if (args.file_path != ""):
    with open(args.file_path, 'w') as output:
        output.write(result)
