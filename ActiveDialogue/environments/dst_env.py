"""In-dialogue selective sampling for slot labeling tasks."""

import random
import logging
import torch
import numpy as np


class DSTEnv():
    """Base class for DST online AL environments."""

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
        self._online_ptrs, self._seed_ptrs, num_turns = self._dataset.get_turn_ptrs(
            args.num_passes, args.seed_size, sample_mode=args.sample_mode)
        logging.info(
            "First 10 dataset stream pointers: %s", self._online_ptrs[:10])
        self.pool_size = args.num_passes * (num_turns - args.seed_size)
        assert self.pool_size == len(self._online_ptrs)
        logging.info("Seed size: %d", len(self._seed_ptrs))
        logging.info("Pool size: %d", len(self._online_ptrs))

        # Inject noise
        if args.noise_fn > 0 or args.noise_fp > 0:
            self._dataset.add_noise(args.noise_fn, args.noise_fp)

        # Load model
        self._model = model_cls(args, self._ontology, vocab)
        self._model.load_emb(Eword)
        self._model = self._model.to(self._model.device)

    def load(self, prefix):
        """Load seeded model for the current seed"""
        success = self._model.load_id("%s %d" % (prefix, self._args.seed))
        if not success:
            return False
        self._model = self._model.to(self._model.device)
        return True

    def save(self, prefix):
        """Save model"""
        self._model.save({}, identifier="%s %d" % (prefix, self._args.seed))

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
        """Return metrics for current model"""
        metrics = {
            "Stream progress":
                self._current_idx / self.pool_size,
            "Exhausted labels proportion":
                self._used_labels / self._args.label_budget,
            "Exhausted labels":
                self._used_labels,
        }

        metrics.update({
            "Labeled / total":
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
        return self._online_ptrs[np.arange(
            self._current_idx,
            min(self._current_idx + self._args.al_batch, self.pool_size))]

    @property
    def can_label(self):
        """Can the environment label more points?"""
        return self._args.label_budget > self._used_labels

    def label_all(self):
        """Fully label all available ptrs"""
        # Add new label ptrs to support ptrs
        self._support_ptrs = np.concatenate(
            [self._support_ptrs,
             self._online_ptrs[self._current_idx:self._current_idx
                               + self._args.label_budget]])
        assert len(np.unique(self._support_ptrs)) == len(self._support_ptrs)
        self._used_labels += self._args.label_budget

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

    def fit(self):
        """Fit on current support and seed datapoints."""
        if self._model.optimizer is None:
            self._model.set_optimizer()

        self._model.train()

        support_iterator = self._dataset.batch(
            batch_size=self._args.batch_size,
            ptrs=np.concatenate([self._support_ptrs, self._seed_ptrs]),
            shuffle=True)
        logging.info("Fitting on %d datapoints.",
                     len(self._support_ptrs) + len(self._seed_ptrs))

        for batch, batch_labels in support_iterator:
            self._model.zero_grad()
            loss, _ = self._model.forward(batch,
                                          batch_labels,
                                          training=True)
            loss.backward()
            self._model.optimizer.step()

    def eval(self):
        """Evaluate underlying model and return metrics."""
        logging.info('Running dev evaluation')
        self._model.eval()
        return self._model.run_eval(self._test_dataset, self._args)
