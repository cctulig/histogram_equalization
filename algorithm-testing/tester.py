import numpy as np

import classifier as c

from pre_processing.blur import blur
from pre_processing.clahe import CLAHE
from pre_processing.grayscale import grayscale
from pre_processing.normalize import normalize

from classification.auto_classifier import auto_trainer, auto_classifier
from classification.ann_classifier import ann_trainer, ann_classifier


def convert_hot_vector(y):
    converted = np.zeros((len(y), np.max(y) - np.min(y) + 1))
    for i in range(len(y)):
        converted[i, y[i]] = 1
    return converted


# Load data
X = np.reshape(np.load("data/example_data/fashion_mnist_images.npy"), (-1, 28, 28))
Y = convert_hot_vector(np.load("data/example_data/fashion_mnist_labels.npy"))

# Test Classifiers
# c.classify_data(X, Y, auto_trainer, [[None]], auto_classifier, [])
c.classify_data(X, Y, ann_trainer, [[5, 20], [64, 256]], ann_classifier, [normalize])