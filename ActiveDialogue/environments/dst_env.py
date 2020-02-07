"""In-dialogue selective sampling for slot labeling tasks."""

import torch
import pdb
import numpy as np
import random
import logging
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
        datasets, self._ontology, vocab, Eword = load_dataset(
        )
        self._dataset = datasets["train"]
        self._test_dataset = datasets["test"]
        self._idxs, seed_idxs, num_turns = self._dataset.get_turn_idxs(
            args.pool_size, args.seed_size, sample_mode=args.sample_mode)

        # Load model
        self._model = model_cls(args, self._ontology, vocab)
        self._model.load_emb(Eword)
        self._model = self._model.to(self._model.device)

        # Support set
        self._used_labels = 0
        self._support_idxs = set()
        self._latest_support_idxs = []
        self._support_masks = {}
        self._support_labels = {}
        for s in self._ontology.slots:
            self._support_masks[s] = np.full(
                (num_turns, len(self._ontology.values[s])),
                fill_value=-1,
                dtype=np.int32)
        for s in self._ontology.slots:
            self._support_labels[s] = np.full(
                (num_turns, len(self._ontology.values[s])),
                fill_value=-1,
                dtype=np.int32)

        # Seed set
        seed_labels = self._dataset.get_labels(seed_idxs)
        self._support_idxs = set(seed_idxs)
        if args.seed_size:
            for s in self._ontology.slots:
                self._support_masks[s][seed_idxs, :] = 1
                self._support_labels[s][seed_idxs, :] = seed_labels[s]

    def train_seed(self):
        self._model = self._model.to(self._model.device)
        self.fit()

        summary = {
            'eval_dev_{}'.format(k): v for k, v in self._model.run_eval(
                self._test_dataset, self._args).items()
        }
        self._model.save(summary, identifier='seed')

    def load_seed(self):
        self._model.load_best_save()
        self._model = self._model.to(self._model.device)

    def observe(self):
        datapoints = self._dataset.batch_idxs()
        scores = []
        labeled_idxs = self._idxs[np.arange(
            self._current_idx, self._current_idx + self._args.al_batch)]
        for batch, _ in self._dataset.batch(batch_size=self._args.batch_size,
                                            idxs=labeled_idxs,
                                            shuffle=False):
            scores += self._model.forward(datapoints)[1]
        return datapoints, scores

    def step(self):
        self._current_idx += self._args.al_batch
        return self._current_idx > self._args.pool_size

    def label(self, label):
        # Grab the turn-idxs of the legal, label turns from this batch
        legal = []
        for s in self._ontology.slots:
            legal.append(np.where(np.any(label[s] != -1, axis=1)))
        label_subidxs = np.where(np.all(np.stack(legal).transpose(1, 0, 2), dim=1))

        if self._args.label_budget < len(label) + self._used_labels:
            label_subidxs = label_subidxs[:max(0, self._args.label_budget -
                                                   self._used_labels)]
        label = label[label_subidxs]
        current_idxs = self._idxs[np.arange(
            self._current_idx, self._current_idx + self._args.al_batch)]
        label_idxs = current_idxs[label_subidxs]
        self._latest_support_idxs = label_idxs

        # Label!
        if label_idxs:
            # Determine feedback
            feedback = True
            for s in self._ontology.slots:
                feedback = feedback and np.all(
                    self._dataset.get_labels(label_idxs)[s][
                        label[s] != -1] == label[s][label[s] != -1])
            # Apply labels
            for s in self._ontology.slots:
                self._support_labels[s][label_idxs][
                    label[s] != -1] = (1 if feedback else 0)
            self._used_labels += len(label)
            self._support_idxs = self._support_idxs.union(set(label_idxs))
            return True
        return False

    def fit(self):
        idxs = self._support_idxs + np.repeat(self._latest_support_idxs,
                                              self._args.recency_bias)

        if self._model.optimizer is None:
            self._model.set_optimizer()

        iteration = 0
        for epoch in range(self._args.epochs):
            logging.info('starting epoch {}'.format(epoch))

            # train and update parameters
            for batch, batch_labels in self._dataset.batch(
                    batch_size=self._args.batch_size,
                    idxs=np.array(idxs, dtype=np.int32),
                    labels={s: np.maximum(v, 0) for s, v in self._support_labels.items()},
                    shuffle=True):
                iteration += 1
                self._model.zero_grad()
                loss, scores = self.forward(batch,
                                            batch_labels,
                                            mask=np.array(batch_labels != -1),
                                            training=True)
                loss.backward()
                self._model.optimizer.step()

    def eval(self):
        logging.info('Running dev evaluation')
        dev_out = self._model.run_eval(self._test_dataset, self._args)
        pprint(dev_out)
