"""Trains a FCNN network. """

# code that actually runs the training data set
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import copy
import random


def insertNoisyLabels(y_data, ratioNoise, seedNoise):
    y_data = np.array(copy.deepcopy(y_data), dtype=np.uint8)
    np.random.seed(seedNoise)
    size_train = len(y_data)
    nNoisy = np.array(np.round(size_train * ratioNoise), dtype=np.int32)
    indNoisy = np.random.permutation(size_train)
    indNoisy = indNoisy[:nNoisy]
    randomLabels = np.random.randint(0, 10, size=(nNoisy, 1), dtype=np.uint8).ravel()
    y_data[indNoisy] = randomLabels
    return y_data

# variables
batch_size = 100
epochs = 30
ratioNoises = [0.0]
data_name='mnist'
model_name='MLP'
seed=1
for ratioNoise in ratioNoises:
# load data
    (train_images, train_labels), (test_images,
                                    test_labels) = datasets.mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_images = tf.image.per_image_standardization(train_images)
    test_images = tf.image.per_image_standardization(test_images)

    train_images = np.reshape(train_images, (train_images.shape[0], -1))
    test_images = np.reshape(test_images, (test_images.shape[0], -1))

    # insert label noise
    if ratioNoise > 0:
        train_labels = insertNoisyLabels(
            y_data=train_labels, ratioNoise=ratioNoise, seedNoise=123456)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(  
        0.05,
        decay_steps=train_labels.size / batch_size,
        decay_rate=0.88,
        staircase=True)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(784,)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(350, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # compile model
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule,momentum=0.98),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])

    # fit data
    history = model.fit(train_images, train_labels,
                        epochs=epochs, batch_size=batch_size, verbose=2)
    # model.summary()
    model.evaluate(test_images,test_labels,batch_size=batch_size,verbose=2)
    model.save(f'./model_{model_name}_{data_name}_noise{ratioNoise}')
