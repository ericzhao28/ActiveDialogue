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
        self._current_idx = 0

        # Select train/test/val dataset split
        datasets, self._ontology, vocab, Eword = load_dataset()
        self._dataset = datasets["train"]
        self._test_dataset = datasets["test"]
        self._ptrs, seed_ptrs, num_turns = self._dataset.get_turn_ptrs(
            args.pool_size, args.seed_size, sample_mode=args.sample_mode)
        assert len(self._ptrs) >= args.pool_size
        self._num_turns = num_turns

        # Load model
        self._model = model_cls(args, self._ontology, vocab)
        self._model.load_emb(Eword)
        self._model = self._model.to(self._model.device)

        # Support set, initialize masks and labels
        self._used_labels = 0
        self._support_masks = {}
        self._support_labels = {}
        self._bag_ptrs = np.array([], dtype=np.int32)
        self._bag_idxs = {}
        self._bag_feedback = np.array([], dtype=np.int32)
        for s in self._ontology.slots:
            self._support_masks[s] = np.zeros(
                (num_turns, len(self._ontology.values[s])), dtype=np.int32)
            self._support_labels[s] = np.zeros(
                (num_turns, len(self._ontology.values[s])), dtype=np.int32)
            self._bag_idxs[s] = []

        # Seed set: grab full labels and load into support set
        seed_labels = self._dataset.get_labels(seed_ptrs)
        self._seed_ptrs = seed_ptrs
        if args.seed_size:
            for s in self._ontology.slots:
                self._support_labels[s][seed_ptrs, :] = seed_labels[s]
                self._support_masks[s][seed_ptrs, :] = 1
            print("Seeding")

            if not self.load_seed():
                self.train_seed()
                print("Saving!")
            else:
                print("Loaded!")
                if self._args.force_seed:
                    self.train_seed()
                    print("Saving!")
                print("Seed metrics", self.metrics(True))

    def train_seed(self):
        """Train a model on seed support and save as seed"""
        self._model = self._model.to(self._model.device)
        self.seed_fit(self._args.seed_epochs)

    def load_seed(self):
        """"Load seeded model for the current seed"""
        success = self._model.load_id('seed' + str(self._args.seed))
        if not success:
            return False
        self._model = self._model.to(self._model.device)
        return True

    def leak_labels(self):
        return self._dataset.get_labels(self.current_ptrs)

    def observe(self):
        """Grab observations and predictive distributions over batch"""
        obs = []
        preds = {}
        self._model.eval()
        for batch, _ in self._dataset.batch(batch_size=self._args.inference_batch_size,
                                            ptrs=self.current_ptrs,
                                            shuffle=False):
            batch_preds = self._model.forward(batch)[1]
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

        total_size = 0
        for s in self._ontology.slots:
            total_size += self._num_turns * len(self._ontology.values[s])
        labeled_size = 0
        for s in self._ontology.slots:
            labeled_size += np.sum(self._support_masks[s])
        metrics.update({"Bit label proportion": labeled_size / total_size})
        metrics.update(
            {"Bag proportion": len(self._bag_ptrs) / self._num_turns})

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

    def label(self, label):
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
                        true_labels[i][np.where(label[s][i])])

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
                    ptrs=self._seed_ptrs,
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
                ptrs=self._seed_ptrs,
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
                        ptrs=self._seed_ptrs,
                        shuffle=True)
                    batch, batch_labels = next(seed_iterator)
                loss, scores = self._model.forward(batch,
                                                   batch_labels,
                                                   training=True)

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

    def eval(self):
        logging.info('Running dev evaluation')
        self._model.eval()
        return self._model.run_eval(self._test_dataset, self._args)
