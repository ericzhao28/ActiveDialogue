"""In-dialogue selective sampling for slot labeling tasks."""

from ActiveDialogue.environments.dst_env import DSTEnv
import torch
import numpy as np
import random
import logging
import pdb
from pprint import pprint


class PartialEnv(DSTEnv):

    def __init__(self, load_dataset, model_cls, args):
        """Initialize environment and cache datasets."""

        super().__init__(load_dataset, model_cls, args)

        # Initialize mask of viewable data points
        self._support_masks = {}
        for s in self._ontology.slots:
            self._support_masks[s] = np.zeros(
                (self._num_turns, len(self._ontology.values[s])),
                dtype=np.int32)

        # All seed items should be fully labeled
        if args.seed_size:
            for s in self._ontology.slots:
                self._support_masks[s][self._support_ptrs, :] = 1
            print("Seeding")

    def metrics(self, run_eval=False):
        metrics = super().metrics(run_eval)

        labeled_bits = 0
        total_bits = 0
        for s in self._ontology.slots:
            labeled_bits += np.sum(self._support_masks[s])
            total_bits += len(self._support_masks[s])

        metrics.update({"Bit label proportion": labeled_bits / total_bits})

        return metrics

    def label(self, label):
        # No more labeling allowed
        if self._args.label_budget <= self._used_labels:
            raise ValueError()

        label = {s: np.array(v, dtype=np.int32) for s, v in label.items()}
        batch_size = len(list(label.values())[0])

        # Grab the turn-idxs of the legal, label turns from this batch:
        # any turn with any non-trivial label
        total_labels = 0
        label_idxs = []
        for i in range(batch_size):
            label_size = 0
            for s in self._ontology.slots:
                # remove redundancy
                redundant = self._support_masks[s][self.current_ptrs[i]] == 1
                label[s][i][redundant] = 0

                label_size += sum(label[s][i])
            if total_labels + label_size + self._used_labels > self._args.label_budget:
                break
            total_labels += label_size
            if label_size:
                label_idxs.append(i)

        # Filter ptrs and labels that are valid
        for s, v in label.items():
            label[s] = label[s][label_idxs]
        label_ptrs = self.current_ptrs[label_idxs]

        # Label!
        if len(label_ptrs):
            for s in self._ontology.slots:
                self._support_masks[s][label_ptrs] = label[s]
            self._support_ptrs = np.concatenate(
                [self._support_ptrs, label_ptrs])
            assert len(self._support_ptrs) == len(
                np.unique(self._support_ptrs))
            self._used_labels += total_labels
            return total_labels

        return 0

    def fit(self, epochs=None, prefix=""):
        # Initialize optimizer and trackers
        if self._model.optimizer is None:
            self._model.set_optimizer()

        iteration = 0
        best = None
        if not epochs:
            epochs = self._args.epochs
        self._model.train()

        for epoch in range(epochs):
            print('Starting fit epoch {}.'.format(epoch))

            # Batch from seed, looping if compound
            seed_iterator = self._dataset.batch(
                batch_size=self._args.comp_batch_size,
                ptrs=self._seed_ptrs,
                shuffle=True,
                loop=True)
            support_iterator = self._dataset.batch(
                batch_size=self._args.batch_size,
                ptrs=self._support_ptrs,
                shuffle=True,
                return_ptrs=True)
            print("Fitting on {} datapoints.".format(len(self._support_ptrs)))

            for batch, batch_labels, batch_ptrs in support_iterator:
                seed_batch, seed_batch_labels = next(seed_iterator)
                self._model.zero_grad()
                loss, _ = self._model.partial_forward(
                    batch,
                    batch_labels,
                    training=True,
                    mask={
                        s: v[batch_ptrs]
                        for s, v in self._support_masks.items()
                    })
                seed_loss, _ = self._model.forward(seed_batch,
                                                   seed_batch_labels,
                                                   training=True)
                (loss + self._args.gamma * seed_loss).backward()
                self._model.optimizer.step()

            # Report metrics, saving if stop metric is best
            metrics = self.metrics(True)
            print("Epoch metrics: ", metrics)
            if best is None or metrics[self._args.stop] > best:
                print("Saving best!")
                self._model.save({}, identifier=prefix + str(self._args.seed))
                best = metrics[self._args.stop]

            self._model.train()
