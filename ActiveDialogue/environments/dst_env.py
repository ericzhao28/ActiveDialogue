"""In-dialogue selective sampling for slot labeling tasks."""

import torch
import numpy as np
import random
import logging
from pprint import pprint

NO_LABEL_IDX = -1
EPSILON = 1e-9


class DSTEnv():

    def __init__(self, dataset_wrapper, model_cls, args):
        """Initialize environment and cache datasets."""

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

        self._args = args
        self._current_idx = 0

        # Select train/test/val dataset split
        datasets, self._ontology, vocab, Eword = dataset_wrapper.load_dataset(
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
        self._support_masks = {}
        self._support_labels = {}
        for s in self._ontology.slots:
            self._support_masks[s] = np.full(
                (num_turns, len(self.ontology.values[s])),
                fill_value=-1,
                dtype=np.int32)
        for s in self._ontology.slots:
            self._support_labels[s] = np.full(
                (num_turns, len(self.ontology.values[s])),
                fill_value=-1,
                dtype=np.int32)

        # Seed set
        seed_labels = self._dataset.get_labels(seed_idxs)
        if args.seed_size:
            for s in self._ontology.slots:
                self._support_masks[s][seed_idxs, :] = 1
                self._support_labels[s][seed_idxs, :] = seed_labels[s]

    def train_seed(self):
        self._model = self._model.to(self._model.device)
        self._fit()

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
        if self._args.label_budget < len(label) + self._used_labels:
            label = label[:max(0, self._args.label_budget -
                               self._used_labels)]
        if label:
            labeled_idxs = self._idxs[np.arange(
                self._current_idx, self._current_idx + self._args.al_batch)]

            feedback = True
            for s in self._ontology.slots:
                feedback = feedback and np.all(
                    self._dataset.get_labels(labeled_idxs)[s][:len(label)][
                        label[s] != -1] == label[s][label[s] != -1])

            for s in self._ontology.slots:
                self._support_labels[labeled_idxs][s][:len(label)][
                    label[s] != -1] = (1 if feedback else 0)
            self._used_labels += len(label)
            return True
        return False

    def fit(self):
        current_idxs = self._idxs[np.arange(
            self._current_idx, self._current_idx + self._args.al_batch)]
        labeled_idxs = np.where(np.all(self._support_labels != -1, axis=1))
        pref_idxs = np.intersect1d(labeled_idxs, current_idxs)
        idxs = current_idxs + np.repeat(pref_idxs, self._args.recency_bias)

        if self._model.optimizer is None:
            self._model.set_optimizer()

        iteration = 0
        for epoch in range(self._args.epochs):
            logging.info('starting epoch {}'.format(epoch))

            # train and update parameters
            for batch, batch_labels in self._dataset.batch(
                    batch_size=self._args.batch_size,
                    idxs=idxs,
                    labels=np.maximum(self._support_labels, 0),
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
