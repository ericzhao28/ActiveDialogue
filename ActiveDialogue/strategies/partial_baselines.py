"""Baseline sampling strategies for partial labeling."""

import numpy as np
import pdb
import random


def random(pred):
    label = {}
    for s, v in pred.items():
        label[s] = np.random.randint(0, 2, v.shape)
    return label


def passive(pred):
    label = {}
    for key in pred.keys():
        label[key] = np.zeros(pred[key].shape)
    return label


def aggressive(pred):
    label = {}
    for key in pred.keys():
        label[key] = np.ones(pred[key].shape)
    return label
