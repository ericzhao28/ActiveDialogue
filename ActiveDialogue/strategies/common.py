"""Implementation of selective sampling strategy categories"""

import numpy as np


class FixedThresholdStrategy():

    def __init__(self, init_threshold, measure_uncertainty):
        self._threshold = init_threshold
        self._measure_uncertainty = measure_uncertainty

    def observe(self, obs):
        value = self._measure_uncertainty(obs)
        if value > self._threshold:
            return

    def update(self, feedback):
        pass


class VariableThresholdStrategy():

    def __init__(self, init_threshold, measure_uncertainty):
        self._threshold = init_threshold
        self._measure_uncertainty = measure_uncertainty

    def observe(self, obs):
        value = self._measure_uncertainty(obs)
        if value > self._threshold:
            return

    def no_op_update(self):
        self._threshold = self._threshold * (1 - self._threshold_scaler)

    def update(self):
        self._threshold = self._threshold * (1 + self._threshold_scaler)


class StochasticVariableThresholdStrategy():

    def __init__(self, init_threshold, measure_uncertainty, noise_std):
        self._threshold = init_threshold
        self._measure_uncertainty = measure_uncertainty
        self._noise_std = noise_std

    def observe(self, obs):
        value = self._measure_uncertainty(obs) * np.random.normal(
            mean=1, std=self._noise)
        if value > self._threshold:
            return

    def no_op_update(self):
        self._threshold = self._threshold * (1 - self._threshold_scaler)

    def update(self):
        self._threshold = self._threshold * (1 + self._threshold_scaler)
