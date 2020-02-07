"""Trivially naive baseline sampling strategies"""

import numpy as np


def random_baseline(env):
    action = env.noop_label()
    action[np.random.randint(env.seq_len())] = np.random.randint(
        env.action_bounds[0], env.action_bounds[1])
    return [action]


def passive_baseline(env):
    return [env.noop_label()]
