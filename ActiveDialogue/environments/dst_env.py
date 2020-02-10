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
        self._test_dataset = datasets["dev"]
        self._ptrs, seed_ptrs, num_turns = self._dataset.get_turn_ptrs(
            args.pool_size, args.seed_size, sample_mode=args.sample_mode)
        assert len(self._ptrs) >= args.pool_size

        # Load model
        self._model = model_cls(args, self._ontology, vocab)
        self._model.load_emb(Eword)
        self._model = self._model.to(self._model.device)

        # Support set, initialize masks and labels
        self._used_labels = 0
        self._support_ptrs = set()
        self._support_masks = {}
        self._support_labels = {}
        for s in self._ontology.slots:
            self._support_masks[s] = np.zeros(
                (num_turns, len(self._ontology.values[s])),
                dtype=np.int32)
            self._support_labels[s] = np.zeros(
                (num_turns, len(self._ontology.values[s])),
                dtype=np.int32)

        # Seed set: grab full labels and load into support set
        seed_labels = self._dataset.get_labels(seed_ptrs)
        self._support_ptrs = set(seed_ptrs)
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

    def train_seed(self):
        """Train a model on seed support and save as seed"""
        self._model = self._model.to(self._model.device)
        self.fit(self._args.seed_epochs)
        self._model.save({}, identifier='seed' + str(self._args.seed))

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
        for batch, _ in self._dataset.batch(batch_size=self._args.batch_size,
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
        preds = {
            s: np.concatenate(v) for s, v in preds.items()
        }
        return obs, preds

    def metrics(self, run_eval=False):
        metrics = {
            "Stream progress": self._current_idx / self._args.pool_size,
            "Exhasuted labels": self._used_labels / self._args.label_budget,
        }
        if run_eval:
            metrics.update(self.eval())
            print(metrics)
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
                self._args.pool_size - 1))]

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
                    feedback[i] = feedback[i] and np.all(true_labels[i][np.where(label[s][i])])
            # Apply labels
            for s in self._ontology.slots:
                for i in range(num_labels):
                    current_label = self._support_labels[s][label_ptrs[i]][np.where(label[s][i])]
                    self._support_labels[s][label_ptrs[i]][np.where(label[s][i])] = np.logical_or(current_label, feedback[i])  # default to positive if not negative 1
                    self._support_masks[s][label_ptrs[i]][np.where(label[s][i])] = 1
            self._used_labels += num_labels
            self._support_ptrs = self._support_ptrs.union(set(label_ptrs))
            return True

        return False

    def fit(self, epochs=None):
        support_ptrs = np.array(list(self._support_ptrs))
        for i in range(len(self._support_ptrs)):
            valid = False
            for s in self._support_masks.keys():
                valid = valid or np.any(self._support_masks[s][support_ptrs[i]])
            assert valid

        if self._model.optimizer is None:
            self._model.set_optimizer()

        iteration = 0
        if not epochs:
            epochs = self._args.epochs
        self._model.train()
        print("Training for {} epochs starting now.".format(epochs))
        for epoch in range(epochs):
            print('starting epoch {}'.format(epoch))

            support_ptrs = np.array(list(self._support_ptrs))
            support_ptrs = support_ptrs[np.random.permutation(np.arange(support_ptrs.shape[0]))][:self._args.fit_items]
            print("Fitting on {} viable turns.".format(len(support_ptrs)))

            # train and update parameters
            for batch, batch_labels, batch_ptrs in self._dataset.batch(
                    batch_size=self._args.batch_size,
                    ptrs=np.array(support_ptrs, dtype=np.int32),
                    labels={
                        s: np.maximum(v, 0, dtype=np.float32)
                        for s, v in self._support_labels.items()
                    },
                    shuffle=True, give_ptrs=True):

                mask = {s: np.array(v[batch_ptrs], dtype=np.float32) for s, v in self._support_masks.items()}
                iteration += 1
                self._model.zero_grad()
                loss, scores = self._model.forward(
                    batch,
                    batch_labels,
                    mask=mask,
                    training=True)
                loss.backward()
                self._model.optimizer.step()

    def eval(self):
        logging.info('Running dev evaluation')
        self._model.eval()
        return self._model.run_eval(self._test_dataset, self._args)
