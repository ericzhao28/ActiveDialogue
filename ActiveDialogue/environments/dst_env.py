"""In-dialogue selective sampling for slot labeling tasks."""

import torch
import numpy as np
import random
import logging
import pdb
from pprint import pprint


class DSTEnv():

    def __init__(self, load_dataset, model_cls, args):
        """Initialize environment and cache datasets."""

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

        self._args = args

        # Position in stream
        self._current_idx = 0
        self._used_labels = 0

        # Select train/test/val dataset split
        datasets, self._ontology, vocab, Eword = load_dataset()
        self._dataset = datasets["train"]
        self._test_dataset = datasets["dev"]
        self._support_ptrs = np.array([], dtype=np.int32)
        self._ptrs, self._seed_ptrs, num_turns = self._dataset.get_turn_ptrs(
            args.pool_size, args.seed_size, sample_mode=args.sample_mode)
        assert len(self._ptrs) >= args.pool_size
        self._num_turns = num_turns

        # Load model
        self._model = model_cls(args, self._ontology, vocab)
        self._model.load_emb(Eword)
        self._model = self._model.to(self._model.device)

    def load_seed(self):
        """Load seeded model for the current seed"""
        success = self._model.load_id('seed' + str(self._args.seed))
        if not success:
            return False
        self._model = self._model.to(self._model.device)
        return True

    def leak_labels(self):
        """Leak ground-truth labels for current stream items"""
        return self._dataset.get_labels(self.current_ptrs)

    def observe(self):
        """Grab observations and predictive distributions over batch"""
        obs = []
        preds = {}
        self._model.eval()
        for batch, _ in self._dataset.batch(
                batch_size=self._args.inference_batch_size,
                ptrs=self.current_ptrs,
                shuffle=False):
            batch_preds = self._model.forward(batch, training=False)[1]
            if not preds:
                for s in batch_preds.keys():
                    preds[s] = []
            for s in batch_preds.keys():
                preds[s].append(batch_preds[s])
            obs.append(batch)
        obs = np.concatenate(obs)
        preds = {s: np.concatenate(v) for s, v in preds.items()}
        return obs, preds

    def metrics(self, run_eval=False):
        metrics = {
            "Stream progress": self._current_idx / self._args.pool_size,
            "Exhasuted labels": self._used_labels / self._args.label_budget,
        }

        metrics.update({
            "Example label proportion":
                len(self._support_ptrs) /
                (self._args.pool_size + self._args.seed_size)
        })
        if run_eval:
            metrics.update(self.eval())

        return metrics

    def step(self):
        """Step forward the current idx in the self._idxs path"""
        self._current_idx += self._args.al_batch
        return self._current_idx >= self._args.pool_size

    @property
    def current_ptrs(self):
        """Expand current_idx into an array of currently occupied idxs"""
        return self._ptrs[np.arange(
            self._current_idx,
            min(self._current_idx + self._args.al_batch,
                self._args.pool_size))]

    @property
    def can_label(self):
        return self._args.label_budget > self._used_labels

    def label(self, label):
        """Fully label ptrs according to list of idxs"""
        # No more labeling allowed
        if self._args.label_budget <= self._used_labels:
            raise ValueError()

        # Get label locations
        label = np.where(label == 1)

        # Filter out redundant label requests
        label = [
            i for i in label if self.current_ptrs[i] not in self._support_ptrs
        ]

        # Limit to label budget
        label = label[:self._args.label_budget - self._used_labels]

        # Add new label ptrs to support ptrs
        self._support_ptrs = np.concatenate(
            [self._support_ptrs, self.current_ptrs[label]])
        assert len(np.unique(self._support_ptrs)) == len(self._support_ptrs)

        self._used_labels += len(label)

        return len(label)

    def seed_fit(self, epochs=None, prefix=""):
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
                batch_size=self._args.seed_batch_size,
                ptrs=self._seed_ptrs,
                shuffle=True,
                loop=False)

            for batch, batch_labels in seed_iterator:
                self._model.zero_grad()
                loss, _ = self._model.forward(batch,
                                              batch_labels,
                                              training=True)
                loss.backward()
                self._model.optimizer.step()

            # Report metrics, saving if stop metric is best
            metrics = self.metrics(True)
            print("Epoch metrics: ", metrics)
            if best is None or metrics[self._args.stop] > best:
                print("Saving best!")
                self._model.save({}, identifier=prefix + str(self._args.seed))
                best = metrics[self._args.stop]

            self._model.train()

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
                shuffle=True)
            print("Fitting on {} datapoints.".format(len(self._support_ptrs)))

            for batch, batch_labels in support_iterator:
                seed_batch, seed_batch_labels = next(seed_iterator)
                self._model.zero_grad()
                loss, _ = self._model.forward(batch,
                                              batch_labels,
                                              training=True)
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

    def eval(self):
        logging.info('Running dev evaluation')
        self._model.eval()
        return self._model.run_eval(self._test_dataset, self._args)
