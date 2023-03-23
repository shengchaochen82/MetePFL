import os
import zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def data_split(sensor_nodes, num_users):
    num_items = int(sensor_nodes/num_users)
    dict_users, all_idxs = {}, [i for i in range(sensor_nodes)]
    print(dict_users, all_idxs)
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def load_metr_la_data(dataset_name):
    if dataset_name == "Percipition":
        Y = np.load("data/Percipition.npy").transpose((2, 0, 1))
    elif dataset_name == "Surface_temp":
        Y = np.load("data/Surface_temp.npy").transpose((2, 0, 1))
    elif dataset_name == "Upstream_flux":
        Y = np.load("data/Upstream_flux.npy").transpose((2, 0, 1))
    Y = Y.astype(np.float32)
    # Normalization using Z-score method
    means = np.mean(Y, axis=(0, 2))
    print(Y.shape)
    Y = Y - means.reshape(1, -1, 1)
    stds = np.std(Y, axis=(0, 2))
    Y = Y / stds.reshape(1, -1, 1)

    return Y, means, stds


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + num_timesteps) for i in range(X.shape[2] - (num_timesteps) + 1)]

    features = []
    for i, j in indices:
        features.append(X[:, :, i: i + num_timesteps].transpose((0, 2, 1)))
    return np.array(features)


# For Federated Learning
def iid_split(sensor_nodes, num_users):
    num_items = int(sensor_nodes/num_users)
    dict_users, all_idxs = {}, [i for i in range(sensor_nodes)]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

