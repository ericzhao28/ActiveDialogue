"""Baseline sampling strategies for binary feedback"""

import numpy as np
import pdb
import random
from ActiveDialogue.utils import split, unsplit


def random_singlets(pred):
    apred, legend = split(pred)
    assert len(apred.shape) == 2
    label = np.zeros_like(apred)
    label_idxs = np.random.randint(0, apred.shape[1], apred.shape[0])
    for i, j in enumerate(label_idxs):
        label[i, j] = 1
    return unsplit(label, list(pred.keys()), legend)


def passive(pred):
    label = {}
    for key in pred.keys():
        label[key] = np.zeros(pred[key].shape)
    return label


def epsilon_cheat(pred, true_labels):
    rnd = random_singlets(pred)
    batch_size = len(list(pred.values())[0])
    for i in range(batch_size):
        if random.random() > 0.8:
            for s in rnd.keys():
                rnd[s][i] = true_labels[s][i]
    return rnd
