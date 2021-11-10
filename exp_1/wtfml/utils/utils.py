import numpy as np


def np_softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))
