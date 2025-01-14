"""This module contains definitions for loading and processing neural networks.

Implements the following concepts:

(1) Load a keras network from disk (or pretrained models) and embed it into a NNModel object that allows easy evaluation and manipulations.

(2) Import of Mnist and Cifar datasets, adding label noise.

(3) Evaluate performance when setting singular values to zero, shifting singular values, filtering neural network weights.

The code refers to the paper https://journals.aps.org/pre/abstract/10.1103/PhysRevE.108.L022302"""

from typing import Union, Dict, Tuple, List
from copy import deepcopy
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import gc


def insert_noisy_labels_mnist(labels: np.ndarray, ratio_noise: float, seed_noise: int) -> np.ndarray:
    """Shuffle a portion of the labels randomly. """
    labels = np.array(deepcopy(labels), dtype=np.uint8)
    np.random.seed(seed_noise)
    size_train = len(labels)
    nNoisy = np.array(np.round(size_train * ratio_noise), dtype=np.int)
    indNoisy = np.random.permutation(size_train)
    indNoisy = indNoisy[:nNoisy]
    randomLabels = np.random.randint(0, 10, size=(nNoisy, 1), dtype=np.uint8).ravel()
    labels[indNoisy] = randomLabels
    return labels

def insert_noisy_labels(labels: np.ndarray, ratio_noise: float, seed_noise: int) -> np.ndarray:
    """Shuffle a portion of the labels randomly. """
    labels = np.array(deepcopy(labels), dtype=np.uint8)
    np.random.seed(seed_noise)
    size_train = len(labels)
    nNoisy = np.array(np.round(size_train * ratio_noise), dtype=np.int)
    indNoisy = np.random.permutation(size_train)
    indNoisy = indNoisy[:nNoisy]
    randomLabels = np.random.randint(0, 10, size=(nNoisy, 1), dtype=np.uint8)
    labels[indNoisy] = randomLabels
    return labels

class NNModel:
    """Neural network model. Model is loaded from a given path,
    datasets as dictionaries with keys like 'test', 'train',... can be assigned and used 
    for evaluation, a Reshaper can be set to automatically get weights in matrix form, 
    and the batch_size can be set which is used for evaluation (relevant only to control 
    GPU memory or for large datasets that do not fit into RAM). 
    """
    model: tf.keras.Sequential
    path: str
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]
    batch_size: int

    def __init__(self, path: Union[str, Path], datasets: Union[None, Dict[str, Tuple[np.ndarray, np.ndarray]]],  batch_size: int = 32):
        """Loads the keras model and initializes the NNModel."""
        self.path = path
        self.datasets = datasets
        self.batch_size = batch_size
        # load keras model
        self.model = self.load_model(path)

    def load_model(self, path: Path) -> tf.keras.Sequential:
        """Load the keras model from file and returns it. """
        model = tf.keras.models.load_model(path)
        model.compile(metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
                               tf.keras.metrics.SparseCategoricalCrossentropy(),
                               tf.keras.metrics.MeanSquaredError()])
        return model

    def get_weight(self, layer_index: int) -> np.ndarray:
        """Gets weight for requested layer"""
        return self.model.layers[layer_index].get_weights()[0]


    def set_weight(self, matrix: np.ndarray, layer_index: int) -> None:
        """Sets "matrix" as the weight of the layer with index "layer_index". """
        weight = deepcopy(matrix)
        weight = [weight, self.model.layers[layer_index].get_weights()[1]]
        self.model.layers[layer_index].set_weights(weight)

    def evaluate(self, dataset_key: str, **kwargs):
        """Evaluates the model's performance on the dataset """
        x, y = self.datasets[dataset_key]
        return self.model.evaluate(x, y, batch_size=self.batch_size, **kwargs)

 
def get_mnist_fc_std(ratio_noise: float = 0.0) -> Dict[str, Tuple[np.ndarray]]:
    """Gets the Mnist dataset for a FCNN network from tensorflow.keras, standardizes the images
    to have unit variance and zero mean, and reshapes them to a matrix of shape
    (number of images, 784). If ratio_noise > 0 label noise (0<=ratio_noise<=1)
    is added to the training dataset by shuffling a ration ratio_noise of the 
    labels randomly with seed 123456."""


    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_images = tf.image.per_image_standardization(train_images)
    test_images = tf.image.per_image_standardization(test_images)

    train_images = np.reshape(train_images, (train_images.shape[0], -1))
    test_images = np.reshape(test_images, (test_images.shape[0], -1))

    # insert label noise
    if ratio_noise > 0:
        train_labels = insert_noisy_labels_mnist(
            labels=train_labels, ratio_noise=ratio_noise, seed_noise=123456)

    return {'train': (train_images, train_labels), 'test': (test_images, test_labels)}

def get_CIFAR10_cnn_std(ratio_noise: float = 0.0) -> Dict[str, Tuple[np.ndarray]]:
    """Gets the CIFAR-10 dataset for a cnn network from tensorflow.keras, standardizes the images
    to have unit variance and zero mean, and leaves the images in their original shape. 
    If ratio_noise > 0 label noise (0<=ratio_noise<=1)
    is added to the training dataset by shuffling a ration ratio_noise of the 
    labels randomly with seed 123456."""

    # load data
    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_images = tf.image.per_image_standardization(train_images)
    test_images = tf.image.per_image_standardization(test_images)

    train_labels = np.array(train_labels.reshape(50000, 1), dtype=np.uint8)
    test_labels = np.array(test_labels.reshape(10000, ), dtype=np.uint8)

    if ratio_noise > 0:
        train_labels = insert_noisy_labels(
            labels=train_labels, ratio_noise=ratio_noise, seed_noise=123456)
    train_labels = train_labels.reshape(50000,)

    return {'train': (train_images, train_labels), 'test': (test_images, test_labels)}

def recover_and_remove_svals(nn_model: NNModel, layer_indices: Union[List[int], int], svals_shifted:np.ndarray,
                           dataset_keys: Union[List[str], str] = ['test', 'train'],) -> Tuple[np.ndarray,Dict[str,np.ndarray],Dict[str,np.ndarray]]:
    """Algorithm 2 """
    accuracies = dict()
    costs = dict()
    for key in dataset_keys:
        accuracies[key] = []
        costs[key] = []
    # get svals and shifted svals
    nn_model_cp = NNModel(nn_model.path, datasets=nn_model.datasets, batch_size=nn_model.batch_size)
    svds = [np.linalg.svd(nn_model_cp.get_weight(
        iLayer), full_matrices=False) for iLayer in layer_indices]

    # shift and remove more and more svals
    nLayers = [np.min(np.shape(nn_model.get_weight(iLayer)))
               for iLayer in layer_indices]
    step = 1 / np.max(nLayers)
    ratioRemoved = np.arange(0.4, 1 + step / 2, step)
    for zop in tqdm(ratioRemoved):
        nRemoves = [int(zop * nLayer) for nLayer in nLayers]
        for iLayer, svd, nRemove in zip(layer_indices, svds, nRemoves):
            if nRemove > 0:
                svals_shifted[-nRemove:] = np.tile(0.0, nRemove)
                newWeight = svd[0] @ np.diag(svals_shifted) @ svd[2]
                nn_model_cp.set_weight(newWeight, iLayer)
                del newWeight
                gc.collect()
        # evaluate after shift and removal
        for key in dataset_keys:
            metrics = nn_model_cp.evaluate(key, verbose=0)
            accuracies[key].append(metrics[1])
            costs[key].append(metrics[2])  

    return ratioRemoved, accuracies, costs



