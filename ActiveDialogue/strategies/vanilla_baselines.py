"""Baseline sampling strategies for fully labeled learning"""

import numpy as np


def random(pred):
    batch_size = len(next(iter(pred.values())))
    return np.random.randint(0, 2, batch_size)


def aggressive(pred):
    return passive(pred) + 1


def passive(pred):
    batch_size = len(next(iter(pred.values())))
    return np.zeros(batch_size)
