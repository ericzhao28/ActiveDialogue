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
        self._indices, seed_indices, num_turns = self._dataset.get_turn_indices(
            args.pool_size, args.seed_size, sample_mode=args._sample_mode)

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
        seed_labels = self._dataset.get_labels(seed_indices)
        if args.seed_size:
            for s in self._ontology.slots:
                self._support_masks[s][seed_indices, :] = 1
                self._support_labels[s][seed_indices, :] = seed_labels[s]

    def train_seed(self):
        self._model = self._model.to(self._model.device)
        self._fit()

        # evalute on train and dev
        summary = {
            'eval_dev_{}'.format(k): v for k, v in self._model.run_eval(
                self._test_dataset, self._args).items()
        }

        # do early stopping saves
        self.save(summary, identifier='seed')

    def load_seed(self):
        self._model.load_best_save()
        self._model = self._model.to(self._model.device)

    def observe(self):
        datapoints = self._dataset.batch_indices()
        loss, scores = self._model.forward(datapoints)
        return datapoints, scores

    def step(self):
        self._current_idx += self._args.al_batch
        return self._current_idx > self._args.pool_size

    def label(self, label):
        feedback = True
        labeled_indices = self._indices(
            np.arange(self._current_idx,
                      self._current_idx + self._args.al_batch))
        for s in self._ontology.slots:
            self._support_labels[labeled_indices][label[s]] = (1 if feedback
                                                               else 0)

    def fit(self, iterations):
        iteration = 0
        labeled_indices = self._indices(
            np.arange(self._current_idx,
                      self._current_idx + self._args.al_batch))

        if self.optimizer is None:
            self.set_optimizer()

        for epoch in range(self._args.epoch):
            logging.info('starting epoch {}'.format(epoch))

            labeled = np.all(self._support_labels != -1, axis=1)
            indices = np.where(labeled)
            supp_indices = np.intersect1d(labeled_indices, indices)
            indices = indices + np.repeat(supp_indices,
                                          self._args.recency_bias)

            # train and update parameters
            for batch, batch_labels in self._dataset.batch(
                    batch_size=self._args.batch_size,
                    indices=indices,
                    labels=np.maximum(self._support_labels, 0),
                    mask=np.array(batch_labels == -1),
                    shuffle=True):
                iteration += 1
                self.zero_grad()
                loss, scores = self.forward(batch, batch_labels)
                loss.backward()
                self.optimizer.step()

    def eval(self):
        logging.info('Running dev evaluation')
        dev_out = self._model.run_eval(self._test_dataset, self._args)
        pprint(dev_out)
