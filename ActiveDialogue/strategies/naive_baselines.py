"""Trivially naive baseline sampling strategies"""

import numpy as np
import pdb
import random


def random_singlets(pred):
    chosen_keys = []
    label = {}
    for i, key in enumerate(pred.keys()):
        if not len(chosen_keys):
            batch_size = len(pred[key])
            chosen_keys = np.array(np.random.randint(0, len(pred.keys()), batch_size))
        idxs = np.where(chosen_keys == i)
        label[key] = np.zeros(pred[key].shape)
        label_idxs = np.random.randint(0, len(pred[key][0]), len(idxs))
        label[key][idxs, label_idxs] = 1
    return label


def passive_baseline(pred):
    label = {}
    for key in pred.keys():
        label[key] = np.zeros(pred[key].shape)
    return label


def epsilon_cheat(pred, true_labels):
    rnd = random_singlets(pred)
    batch_size = len(list(pred.values())[0])
    for i in range(batch_size):
        if random.random() > 0.5:
            for s in rnd.keys():
                rnd[s][i] = true_labels[s][i]
    return rnd


