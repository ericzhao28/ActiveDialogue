"""In-dialogue selective sampling for slot labeling tasks."""

import torch
from prettytable import PrettyTable
import numpy as np

NO_LABEL_IDX = -1
EPSILON = 1e-9


class LabelerEnv():

  def __init__(self, dataset_wrapper, tokenizer, retokenization, test, device="cpu",
               precap=None, max_seq_len=12):
    """Initialize environment and cache datasets."""

    # Select train/test/val dataset split
    (self._train_dataset, self._test_dataset,
     self._val_dataset) = dataset_wrapper.preprocess_dataset(
         tokenizer, retokenization, max_seq_len=max_seq_len)
    if test:
      dataset_split = self._test_dataset
    else:
      dataset_split = self._train_dataset

    # Store dataset supplementary information
    self._downstream_slots = lambda i: dataset_split[
        "supp"]["slots_downstream"][i]
    self._upstream_slots = lambda i: dataset_split[
        "supp"]["slots_upstream"][i]
    self._slots_count = dataset_split["supp"]["slots_count"]
    self._example_count = dataset_split["supp"]["example_count"]
    self._slots_map = dataset_split["supp"]["slots_map"]
    self._atomic_mask = np.full((self._slots_count,),
                                fill_value=0,
                                dtype=np.int32)
    for slot_i in range(self._slots_count):
      if len(self._downstream_slots(slot_i)) == 1:
        self._atomic_mask[slot_i] = 1

    # Select components from our dataset split
    self._base_dataset = {}
    for k in ["encoded_slot_idxs", "slot_idxs", "tokens", "encoded_tokens"]:
      self._base_dataset[k] = np.array(dataset_split[k])

    # Pre-cap dataset size (reduce overhead of caching dataset reps)
    if precap and precap < self._example_count:
      indices = np.random.permutation(
          np.arange(self._example_count))[:precap]
      for k in self._base_dataset.keys():
        self._base_dataset[k] = self._base_dataset[k][indices]
      self._example_count = precap

    # Stream properties
    self._pool_size = None
    self._max_seq_len = max_seq_len
    self._pot_labels = None
    self._episode_ended = None
    self._current_idx = None
    self._prev_guesses = None
    self._total_labels = None
    self._used_labels = None
    self._total_multi_questions = None
    self._used_multi_questions = None
    self._fit_period = None

    # Support set properties
    self._seed_size = None
    self._support_indices = None
    self._support_supp = None

    # Inner-model state
    self._batch_size = None
    self._iterations = None
    self._device = device
    self._init(device)

  def _init(self, device="cpu"):
    raise NotImplementedError()

  def reset(self,
            sample_mode,
            pool_size,
            budget,
            seed_size,
            batch_size=32,
            iterations=50,
            question_cap=2,
            fit_period=10):
    """Initialize selective sampling task."""

    # Reset cache
    self._model = None
    self._posterior_model = None
    torch.cuda.empty_cache()

    ##################################################
    ##### Initialize the stream

    # Record stream properties
    self._pool_size = pool_size
    self._seed_size = seed_size
    self._episode_ended = False
    self._current_idx = 0

    # Sample from dataset
    if sample_mode == "singlepass":
      indices = np.random.permutation(
          np.arange(0, self._example_count))[:pool_size]
    elif sample_mode == "uniform":
      indices = np.random.randint(0, self._example_count, shape=(pool_size,))
    elif sample_mode == "normal":
      raise NotImplementedError()
    else:
      raise ValueError("ClassificationEnv: Invalid sample mode")

    # Build stream dataset from base dataset
    self._dataset = {}
    for k in self._base_dataset.keys():
      self._dataset[k] = self._base_dataset[k][indices]

    # Previous guesses
    self._prev_guesses = np.full((pool_size, self._max_seq_len),
                                 fill_value=-1, dtype=np.int32)
    self._pot_labels = np.full(
        (self._pool_size, self._max_seq_len, self._slots_count),
        fill_value=1, dtype=np.int32)

    # Initialize stream state
    self._total_labels = budget
    self._used_labels = 0
    self._total_multi_questions = question_cap
    self._used_multi_questions = 0

    # Initialize support set
    self._support_indices = np.full(self._total_labels, fill_value=-1,
                                    dtype=np.int32)
    self._support_supp = {}
    self._support_supp["labels"] = np.full((
        self._total_labels, self._max_seq_len), fill_value=-1, dtype=np.int32)
    self._support_supp["feedback"] = np.full(self._total_labels,
                                             fill_value=False, dtype=np.bool)

    self._fit_period = fit_period
    self._batch_size = batch_size
    self._iterations = iterations

    # Train inner-model on support set
    if self._seed_size:
      assert self._seed_size <= self._total_labels
      for _ in range(self._seed_size):
        # Update support set
        self._support_indices[self._used_labels] = self._current_idx
        self._pot_labels[self._current_idx,
                         :self.seq_len(self._current_idx)] = \
            self._pot_labels[self._current_idx,
                             :self.seq_len(self._current_idx)] & \
            self._dataset["encoded_slot_idxs"][self._current_idx]
        self._used_labels += 1
        self._current_idx += 1

    self._reset()

    return -1, self.eval(), False

  def _reset(self):
    raise NotImplementedError()

  def eval(self):
    raise NotImplementedError()

  def _predict(self):
    raise NotImplementedError()

  def _query_label(self, label):
    """Add a label for the current example to support set.

    Updates support set trackers and potential labels accordingly."""
    assert len(label) == self.seq_len()

    # Grab all labels for which we should answer yes
    feedback = True
    for j in range(self.seq_len()):
      if label[j] == NO_LABEL_IDX:
        continue
      upstream_labels = self._upstream_slots(
          self._dataset["slot_idxs"][self._current_idx][j])
      feedback = feedback and label[j] in upstream_labels

    # Update support set
    self._support_indices[self._used_labels] = self._current_idx
    self._support_supp["labels"][self._used_labels][:self.seq_len()] = label
    self._support_supp["feedback"][self._used_labels] = feedback

    # Update inner-model labels (support_pot_labels)
    if feedback:
      # If positive feedback, rule out all slots not downstream of the label
      new_label = np.full((self._max_seq_len, self._slots_count,),
                          fill_value=1, dtype=np.int32)
      for i in range(self.seq_len()):
        if label[i] == NO_LABEL_IDX:
          continue
        new_label[i] = 0
        new_label[i][self._downstream_slots(label[i])] = 1
    else:
      # If negative feedback, assume no information gain
      new_label = np.full((self._max_seq_len, self._slots_count,),
                          fill_value=1, dtype=np.int32)

    # Take bit-wise AND over the integer operands to apply restrictions
    self._pot_labels[self._current_idx] = self._pot_labels[
        self._current_idx] & new_label

  def step(self, action):
    # Terminate if stream is already concluded
    if self._episode_ended:
      raise StopIteration()

    for i, a in enumerate(action[:self.seq_len()]):
      if isinstance(a, str):
        action[i] = self._slots_map[a]

    action = np.array(action)[:self.seq_len()]

    # If labeling action is selected and legal...
    if np.any(action != NO_LABEL_IDX) and \
        self._used_labels < self._total_labels and \
        self._used_multi_questions < self._total_multi_questions:
      # Query for a given label (stepping down the requested label by 1 to
      # account for NO_LABEL_IDX action)
      self._query_label(action[:self.seq_len()])
      self._used_labels += 1
      if self._used_labels % self._fit_period == 0:
        # Depends on label_used, so must occur after increment
        self._fit_model(batch_size=self._batch_size,
                        iterations=self._iterations)
      self._used_multi_questions += 1

    else:
      # Store reaction to stream item
      predicted_label = self._predict()
      self._prev_guesses[self._current_idx] = predicted_label

      # Record if stream has concluded, return final eval.
      if self._current_idx == self._pool_size - 1:
        assert np.sum(self._prev_guesses[self._seed_size:] == -1) == 0

        reward = 0
        for i, predicted_labels in enumerate(
            self._prev_guesses[self._seed_size:]):
          success = True
          for j in range(self.seq_len(i)):
            success = success and predicted_labels[j] == \
                self._dataset["slot_idxs"][i][j]
          reward += int(success)
        self._episode_ended = True
        self._fit_model(batch_size=self._batch_size,
                        iterations=self._iterations)
        return float(reward) / self._pool_size, self.eval(), True

      # Move to next item
      self._used_multi_questions = 0
      self._current_idx += 1

    return -1, -1, False

  def display(self):
    table = PrettyTable()
    table.field_names = ["Query", "Partial label", "Correctness"]
    for i in range(self._current_idx):
      if i in self._support_indices:
        for label_i in np.where(self._support_indices == i)[0]:
          if self._support_supp["labels"][label_i][0] == -1:
            table.add_row(row=[
                "".join(self._dataset["tokens"][label_i]),
                "Support",
                None])
          else:
            table.add_row(row=[
                "".join(self._dataset["tokens"][label_i]),
                " ".join([self._slots_map[x] for x in
                          self._support_supp[
                              "labels"][label_i][:self.seq_len(label_i)]]),
                self._support_supp["feedback"][label_i]
            ])

    print(table)
    print(self._pot_labels)
    print("Label budget: {} / {}".format(self._used_labels,
                                         self._total_labels))
    print("Stream progress: {} / {}".format(self._current_idx,
                                            self._pool_size))
    print("Current score: {}".format(self.eval()))
    print("Current query: {}".format(
        "".join(self._dataset["tokens"][self._current_idx])))
    print("Current true label: {}".format(
        " ".join([self._slots_map[i] for i in
                  self._dataset["slot_idxs"][self._current_idx]])))
    print("Options: {}".format(
        ", ".join([self._slots_map[i] for i in range(self._slots_count)])))

  # Helpful properties and accessors for defining strategies

  @property
  def action_bounds(self):
    """Random, potentially redundant action."""
    return [0, self._slots_count]

  def true_label(self):
    return self._dataset["slot_idxs"][self._current_idx]

  def noop_label(self):
    return np.ones_like(self._dataset["slot_idxs"][self._current_idx]) \
        * NO_LABEL_IDX

  @property
  def dataset(self):
    return self._dataset

  @property
  def downstream_slots(self):
    return self._downstream_slots

  def mask(self, i=None):
    if i is None:
      i = self._current_idx
    mask = self._atomic_mask * self._pot_labels[i]
    mask[self.seq_len(i):] = 0
    return mask

  def seq_len(self, i=None):
    if i is None:
      i = self._current_idx
    return len(self._dataset["tokens"][i])

