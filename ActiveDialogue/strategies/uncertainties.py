import numpy as np


def lc(pred):
    return np.max(partial_lc(pred), axis=1)


def bald(preds):
    return np.max(partial_bald(pred), axis=1)


def partial_lc(pred):
    return np.min(np.array([pred, 1 - pred]), axis=0)


def partial_bald(preds):
    disag = np.sum(preds > 0.5, axis=0)
    return np.max(disag, preds.shape[0] - disag)


def lc_singlet(pred):
    c = partial_lc(pred)
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
