import numpy as np


def split(x):
    data = []
    keys = sorted(list(x.keys()))
    data = [x[k] for k in keys]
    legend = [0]
    for k in keys:
        legend.append(legend[-1] + x[k].shape[-1])
    print(legend)
    return np.concatenate(data, axis=-1), legend

def unsplit(x, keys, legend):
    data = {}
    keys = sorted(keys)
    for i, k in enumerate(keys):
        data[k] = x[legend[i]:legend[i+1]]
    return data

