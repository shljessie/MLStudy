""""
Regression: MSE, MAE, Huber Loss.
Binary Classification: Binary Cross-Entropy, Hinge Loss.
Multiclass Classification: Cross-Entropy.
Probabilistic Predictions: KL Divergence.
Image Segmentation: Dice Loss.

"""


import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]


def cosine_similarity_loss(y_true, y_pred):
    dot_product = np.sum(y_true * y_pred, axis=1)
    norm_true = np.linalg.norm(y_true, axis=1)
    norm_pred = np.linalg.norm(y_pred, axis=1)
    cosine_similarity = dot_product / (norm_true * norm_pred)
    return np.mean(1 - cosine_similarity)

def kl_divergence(y_true, y_pred):
    epsilon = 1e-15
    y_true = np.clip(y_true, epsilon, 1)
    y_pred = np.clip(y_pred, epsilon, 1)
    return np.sum(y_true * np.log(y_true / y_pred))
