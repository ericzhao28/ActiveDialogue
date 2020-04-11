import numpy as np


def split(x):
    def sub_split(x):
        keys = sorted(list(x.keys()))
        data = [x[k] for k in keys]
        legend = [0]
        for k in keys:
            legend.append(legend[-1] + x[k].shape[-1])
        return np.concatenate(data, axis=-1), legend

    if isinstance(x, list):
        return [sub_split(y)[0] for y in x], sub_split(x[0])[1]
    else:
        return sub_split(x)


def unsplit(x, keys, legend):
    assert len(x.shape) == 2
    data = {}
    keys = sorted(keys)
    for i, k in enumerate(keys):
        data[k] = x[..., legend[i]:legend[i + 1]]
    return data
