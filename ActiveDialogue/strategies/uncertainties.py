import numpy as np
from collections import Counter


def mode_rows(a):
    a = np.ascontiguousarray(a)
    void_dt = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    _, ids, count = np.unique(a.view(void_dt).ravel(),
                              return_index=1, return_counts=1)
    largest_count_id = ids[count.argmax()]
    most_frequent_row = a[largest_count_id]
    return most_frequent_row, count.max()


def entropy(pred):
    pred = np.array(pred)
    return np.sum(partial_entropy(pred), axis=1)


def bald(preds):
    preds = np.array(preds).transpose([1, 0, 2]) > 0.5
    results = []
    for p in preds:
        results.append(1 - mode_rows(p)[1] / len(p))
    return np.array(results)


def partial_entropy(pred):
    return -1 * (pred * np.log(pred + 1e-8) +
                 (1 - pred) * np.log(1 - pred + 1e-8))


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
