"""In-dialogue selective sampling for slot labeling tasks."""

from ActiveDialogue.environments.dst_env import DSTEnv
import torch
import numpy as np
import random
import logging
import pdb
from pprint import pprint


class BagEnv(DSTEnv):

    def __init__(self, load_dataset, model_cls, args):
        """Initialize environment and cache datasets."""

        super().__init__(load_dataset, model_cls, args)

        # Support set, initialize masks and labels
        self._bag_ptrs = np.array([], dtype=np.int32)
        self._bag_idxs = {}
        self._bag_feedback = np.array([], dtype=np.int32)
        for s in self._ontology.slots:
            self._bag_idxs[s] = []

    def metrics(self, run_eval=False):
        metrics = super().metric(run_eval)

        total_size = 0
        for s in self._ontology.slots:
            total_size += self._num_turns * len(self._ontology.values[s])
        labeled_size = 0
        for s in self._ontology.slots:
            labeled_size += np.sum(self._support_masks[s])
        metrics.update({"Bit label proportion": labeled_size / total_size})
        metrics.update(
            {"Bag proportion": len(self._bag_ptrs) / self._num_turns})

        return metrics

    def confirmation_label(self, label):
        """Label current batch of data"""
        label = {s: np.array(v, dtype=np.int32) for s, v in label.items()}
        batch_size = len(list(label.values())[0])

        # No more labeling allowed
        if self._args.label_budget <= self._used_labels:
            return False

        # Grab the turn-idxs of the legal, label turns from this batch:
        # any turn with any non-trivial label
        label_idxs = []
        for i in range(batch_size):
            for s in self._ontology.slots:
                if np.any(label[s][i]):
                    label_idxs.append(i)
                    break

        # Cut label idxs by remaining label budget...
        num_labels = len(label_idxs)
        if self._args.label_budget < num_labels + self._used_labels:
            label_idxs = label_idxs[:max(
                0, self._args.label_budget - self._used_labels)]
        num_labels = len(label_idxs)

        # Grab ptrs and labels that are valid
        for s, v in label.items():
            label[s] = label[s][label_idxs]
        label_ptrs = self.current_ptrs[label_idxs]

        # Label!
        if len(label_ptrs):
            # Determine feedback: assume label guess is correct. For each
            # query and slot, check that all values corresponding to label
            # request have positive label.
            feedback = np.full(num_labels, fill_value=1, dtype=np.int32)
            for s in self._ontology.slots:
                true_labels = self._dataset.get_labels(label_ptrs)[s]
                for i in range(num_labels):
                    feedback[i] = feedback[i] and np.all(
                        true_labels[i][np.where(label[s][i])] == 1)

            for i in range(num_labels):
                totals = 0
                for s in self._ontology.slots:
                    self._bag_idxs[s].append(np.where(label[s][i])[0])
                    totals += len(self._bag_idxs[s][-1])
                assert totals > 0
                if totals == 1 or feedback[i] == 1:
                    self._support_masks[s][label_ptrs[i]][np.where(
                        label[s][i])] = 1

            self._bag_feedback = np.concatenate(
                [self._bag_feedback, feedback])
            self._bag_ptrs = np.concatenate([self._bag_ptrs, label_ptrs])
            self._used_labels += num_labels

            return True

        return False

    def seed_fit(self, epochs=None):
        if self._model.optimizer is None:
            self._model.set_optimizer()

        iteration = 0
        best = None
        if not epochs:
            epochs = self._args.epochs
        self._model.train()

        print("Seed training for {} epochs starting now.".format(epochs))
        for epoch in range(epochs):
            print('starting epoch {}'.format(epoch))

            for batch, batch_labels in self._dataset.batch(
                    batch_size=self._args.batch_size,
                    ptrs=self._support_ptrs,
                    shuffle=True):

                iteration += 1
                print("Iteration: ", iteration)
                self._model.zero_grad()

                loss, scores = self._model.forward(batch,
                                                   batch_labels,
                                                   training=True)
                loss.backward()
                self._model.optimizer.step()

            metrics = self.metrics(True)
            print(metrics)
            if best is None or metrics[self._args.stop] > best:
                print("Saving best!")
                self._model.save({}, identifier='seed' + str(self._args.seed))
                best = metrics[self._args.stop]
            self._model.train()

    def fit(self, epochs=None):
        if self._model.optimizer is None:
            self._model.set_optimizer()

        iteration = 0
        if not epochs:
            epochs = self._args.epochs
        self._model.train()

        print("Training for {} epochs starting now.".format(epochs))
        for epoch in range(epochs):
            print('starting epoch {}'.format(epoch))

            seed_iterator = self._dataset.batch(
                batch_size=self._args.batch_size,
                ptrs=self._support_ptrs,
                shuffle=True)

            shuffled_bag_idxs = np.random.permutation(
                np.arange(len(self._bag_ptrs)))[:self._args.fit_items]
            print("Fitting on {} bags".format(len(shuffled_bag_idxs)))

            for batch_i, (bag_batch, _) in enumerate(
                    self._dataset.batch(
                        batch_size=self._args.bag_batch_size,
                        ptrs=self._bag_ptrs[shuffled_bag_idxs])):
                print("Batch: ", batch_i + 1)

                iteration += 1
                self._model.zero_grad()

                try:
                    batch, batch_labels = next(seed_iterator)
                except StopIteration:
                    seed_iterator = self._dataset.batch(
                        batch_size=self._args.batch_size,
                        ptrs=self._support_ptrs,
                        shuffle=True)
                    batch, batch_labels = next(seed_iterator)
                loss, scores = self._model.forward(batch,
                                                   batch_labels,
                                                   training=True)

                for bag_i in shuffled_bag_idxs[batch_i * self._args.bag_batch_size:
                        (batch_i + 1) * self._args.bag_batch_size]:
                    bag_feedback = self._bag_feedback[bag_i]
                    for s in self._bag_idxs.keys():
                        bag_label = self._dataset.get_labels([self._bag_ptrs[bag_i]])[s][0]
                        for label_i in self._bag_idxs[s][bag_i]:
                            assert(bag_feedback == bag_label[label_i])

                bloss, bscores = self._model.bag_forward(
                    bag_batch, {
                        s:
                        np.array(v)[shuffled_bag_idxs[batch_i *
                                                      self._args.bag_batch_size:
                                                      (batch_i + 1) *
                                                      self._args.bag_batch_size]]
                        for s, v in self._bag_idxs.items()
                    },
                    self._bag_feedback[
                        shuffled_bag_idxs[batch_i *
                                          self._args.bag_batch_size:(batch_i +
                                                                 1) *
                                          self._args.bag_batch_size]],
                    training=True,
                    sl_reduction=self._args.sl_reduction,
                    optimistic_weighting=self._args.optimistic_weighting)

                (self._args.gamma * loss + bloss).backward()
                self._model.optimizer.step()

            metrics = self.metrics(True)
            print(metrics)
            self._model.train()
