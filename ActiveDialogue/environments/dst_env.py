"""In-dialogue selective sampling for slot labeling tasks."""

import torch
import numpy as np
import random
import logging
import pdb


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
            args.num_passes, args.seed_size, sample_mode=args.sample_mode)
        self._num_turns = num_turns
        self.pool_size = args.num_passes * (self._num_turns - args.seed_size)
        assert self.pool_size == len(self._ptrs)
        logging.debug("Seed size: {}".format(len(self._seed_ptrs)))
        logging.debug("Pool size: {}".format(len(self._ptrs)))

        # Inject noise
        if args.noise_fn > 0 or args.noise_fp > 0:
            self._dataset.add_noise(args.noise_fn, args.noise_fp)

        # Load model
        def load_model():
            self._model = model_cls(args, self._ontology, vocab)
            self._model.load_emb(Eword)
            self._model = self._model.to(self._model.device)

        load_model()
        self._reset_model = load_model

    def id(self):
        return "seed_{}_strat_{}_noise_fn_{}_noise_fp_{}_num_passes_{}_seed_size_{}_model_{}_batch_size_{}_gamma_{}_label_budget_{}_epochs_{}".format(
            self._args.seed, self._args.strategy, self._args.noise_fn,
            self._args.noise_fp, self._args.num_passes, self._args.seed_size,
            self._args.model, self._args.batch_size, self._args.gamma,
            self._args.label_budget, self._args.epochs)

    def load(self, prefix):
        """Load seeded model for the current seed"""
        success = self._model.load_id(prefix + str(self._args.seed))
        if not success:
            return False
        self._model = self._model.to(self._model.device)
        return True

    def leak_labels(self):
        """Leak ground-truth labels for current stream items"""
        return self._dataset.get_labels(self.current_ptrs)

    def observe(self, num_preds=1):
        """Grab observations and predictive distributions over batch"""
        obs = []
        all_preds = [{} for _ in range(num_preds)]

        if num_preds == 1:
            self._model.eval()
        else:
            self._model.train()

        for batch, _ in self._dataset.batch(
                batch_size=self._args.inference_batch_size,
                ptrs=self.current_ptrs,
                shuffle=False):
            obs.append(batch)

            for i in range(num_preds):
                preds = all_preds[i]
                batch_preds = self._model.forward(batch)[1]
                if not preds:
                    for s in batch_preds.keys():
                        preds[s] = []
                for s in batch_preds.keys():
                    preds[s].append(batch_preds[s])

        obs = np.concatenate(obs)
        all_preds = [{s: np.concatenate(v)
                      for s, v in preds.items()}
                     for preds in all_preds]
        return obs, all_preds

    def metrics(self, run_eval=False):
        metrics = {
            "Stream progress":
                self._current_idx / self.pool_size,
            "Exhausted label budget":
                self._used_labels / self._args.label_budget,
            "Exhausted labels":
                self._used_labels,
        }

        metrics.update({
            "Example label proportion":
                len(self._support_ptrs) /
                (self.pool_size + self._args.seed_size)
        })
        if run_eval:
            metrics.update(self.eval())

        return metrics

    def step(self):
        """Step forward the current idx in the self._idxs path"""
        self._current_idx += self._args.al_batch
        return self._current_idx >= self.pool_size

    @property
    def current_ptrs(self):
        """Expand current_idx into an array of currently occupied idxs"""
        return self._ptrs[np.arange(
            self._current_idx,
            min(self._current_idx + self._args.al_batch, self.pool_size))]

    @property
    def can_label(self):
        return self._args.label_budget > self._used_labels

    def label(self, label):
        """Fully label ptrs according to list of idxs"""
        # No more labeling allowed
        if self._args.label_budget <= self._used_labels:
            raise ValueError()

        # Get label locations
        label = np.where(label == 1)[0]  # FIXED: where returns a tuple?

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
        best = self.metrics(True)
        if not epochs:
            epochs = self._args.epochs
        self._model.train()

        for epoch in range(epochs):
            logging.debug('Starting fit epoch {}.'.format(epoch))

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
            logging.debug("Epoch metrics: {}".format(metrics))
            if metrics[self._args.stop] > best[self._args.stop]:
                logging.debug("Saving best!")
                self._model.save({}, identifier=prefix + str(self._args.seed))
                best = metrics

            self._model.train()
        return best

    def fit(self, epochs=None, prefix="", reset_model=False):
        # Reset model if necessary
        if reset_model:
            self._reset_model()
            self.load('seed')

        # Initialize optimizer and trackers
        if self._model.optimizer is None:
            self._model.set_optimizer()

        iteration = 0
        best = None
        if not epochs:
            epochs = self._args.epochs
        self._model.train()

        for epoch in range(epochs):
            logging.debug('Starting fit epoch {}.'.format(epoch))

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
            logging.debug("Fitting on {} datapoints.".format(
                len(self._support_ptrs)))

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
            logging.debug("Epoch metrics: {}".format(metrics))
            if best is None or metrics[self._args.stop] > best[
                    self._args.stop]:
                logging.debug("Saving best!")
                self._model.save({}, identifier=prefix + str(self._args.seed))
                best = metrics

            self._model.train()
        return best

    def eval(self):
        logging.debug('Running dev evaluation')
        self._model.eval()
        return self._model.run_eval(self._test_dataset, self._args)
