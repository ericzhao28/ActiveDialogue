import numpy as np
from collections import Counter


def entropy(pred):
    return np.sum(partial_entropy(pred), axis=1)


def bald(preds):
    return 1 - Counter(map(tuple, preds)).most_common()[1] / len(preds)


def partial_entropy(pred):
    return pred * np.log(pred) + (1 - pred) * np.log(1 - pred)


def partial_bald(preds):
    disag = np.sum(np.array(preds) > 0.5, axis=0) / len(preds)
    return np.min([disag, 1 - disag], axis=1)


def entropy_singlet(pred):
    c = partial_entropy(pred)
    mask = np.array(np.ones_like(c), dtype=np.bool_)
    mask[np.argmax(c, axis=1)] = False
    c[mask] = 0
    return c


def bald_singlet(preds):
    c = partial_bald(preds)
    mask = np.array(np.ones_like(c), dtype=np.bool_)
    mask[np.argmax(c, axis=1)] = False
    c[mask] = 0
    return c
