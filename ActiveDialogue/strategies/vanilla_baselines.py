"""Baseline sampling strategies for fully labeled learning"""

import numpy as np


def random(pred):
    return np.random.randint(0, 2, len(pred))

def aggressive(pred):
    return np.zeros(len(pred)) + 1

def passive(pred):
    return np.zeros(len(pred))
