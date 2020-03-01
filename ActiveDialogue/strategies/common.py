"""Implementation of selective sampling strategy categories"""

import numpy as np
from ActiveDialogue.utils import split, unsplit


class ThresholdStrategy():
    def __init__(self, measure_uncertainty, args):
        self._measure_uncertainty = measure_uncertainty
        self._threshold = args.init_threshold
        self._threshold_scaler = args.threshold_scaler
        self._noise_std = args.threshold_noise_std

    def observe(self, obs):
        aobs, legend = split(obs)
        value = self._measure_uncertainty(aobs) > self.threshold
        value = unsplit(value, list(obs.keys()), legend)
        return value

    @property
    def threshold(self):
        pass

    def update(self, feedback):
        pass

    def no_op_update(self):
        pass

class FixedThresholdStrategy(ThresholdStrategy):

    @property
    def threshold(self):
        return self._threshold


class VariableThresholdStrategy(ThresholdStrategy):

    @property
    def threshold(self):
        return self._threshold

    def no_op_update(self):
        self._threshold = self._threshold * (1 - self._threshold_scaler)

    def update(self, n=1):
        for i in range(n):
            self._threshold = self._threshold * (1 + self._threshold_scaler)


class StochasticVariableThresholdStrategy(ThresholdStrategy):

    @property
    def threshold(self):
        return self._threshold * np.random.normal(
            mean=1, std=self._noise)

    def no_op_update(self):
        self._threshold = self._threshold * (1 - self._threshold_scaler)

    def update(self, n=1):
        for i in range(n):
            self._threshold = self._threshold * (1 + self._threshold_scaler)
