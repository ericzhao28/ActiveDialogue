"""Implementation of selective sampling strategy categories"""

import numpy as np
from ActiveDialogue.utils import split, unsplit
import math


class ThresholdStrategy():

    def __init__(self, measure_uncertainty, args, need_unsplit=True):
        self._measure_uncertainty = measure_uncertainty
        self._threshold = args.init_threshold
        self._threshold_scaler = args.threshold_scaler
        self._noise_std = args.threshold_noise_std
        self._rejection_ratio = args.rejection_ratio
        self._need_unsplit = need_unsplit

    def observe(self, obs):
        aobs, legend = split(obs)
        value = self._measure_uncertainty(aobs)
        value = value > self.threshold
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
        self._threshold = self._threshold * math.pow(
            1 + self._threshold_scaler, n)
        self._threshold = self._threshold * math.pow(
            1 + self._threshold_scaler,
            float(m - n) / self._rejection_ratio)


class StochasticVariableThresholdStrategy(VariableThresholdStrategy):

    @property
    def threshold(self):
        return self._threshold * np.random.normal(loc=1,
                                                  scale=self._noise_std)
