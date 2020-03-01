"""Implementation of selective sampling strategy categories"""

import numpy as np
from ActiveDialogue.utils import split, unsplit


class ThresholdStrategy():
    def __init__(self, measure_uncertainty, args, need_unsplit=True):
        self._measure_uncertainty = measure_uncertainty
        self._threshold = args.init_threshold
        self._threshold_scaler = args.threshold_scaler
        self._noise_std = args.threshold_noise_std
        self._need_unsplit = need_unsplit

    def observe(self, obs):
        aobs, legend = split(obs)
        if True:
            print(self._measure_uncertainty(aobs))
            print(self.threshold)
        value = self._measure_uncertainty(aobs) > self.threshold
        value = np.array(value, dtype=np.int32)
        if self._need_unsplit:
            value = unsplit(value, list(obs.keys()), legend)
        return value

    @property
    def threshold(self):
        pass

    def update(self, n, m):
        pass

class FixedThresholdStrategy(ThresholdStrategy):

    @property
    def threshold(self):
        return self._threshold


class VariableThresholdStrategy(FixedThresholdStrategy):

    def update(self, n, m):
        print(self._threshold)
        print(n, m)
        for i in range(n):
            self._threshold = self._threshold * (1 + self._threshold_scaler)
        if m - n > 0:
            for i in range(m - n):
                self._threshold = self._threshold * (1 - self._threshold_scaler)
        print(self._threshold)


class StochasticVariableThresholdStrategy(VariableThresholdStrategy):

    @property
    def threshold(self):
        return self._threshold * np.random.normal(
            mean=1, std=self._noise)
