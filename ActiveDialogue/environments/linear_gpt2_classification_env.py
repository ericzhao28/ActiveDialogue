"""In-dialogue selective sampling for intent classification tasks."""

from ActiveDialogue.models.linear_gpt2_classifier import cached_collate_fn, \
    ClassificationHead, atom_nll_loss, Dataset, collate_fn
from ActiveDialogue.environment.classification_env import ClassificationEnv

import torch
import torch.optim as optim
from scipy.special import softmax
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import math

NO_LABEL_IDX = 0
EPSILON = 1e-9


class LinearGPT2ClassificationEnv(ClassificationEnv):

  def _init(self, device="cpu"):
    # Place holder for our models
    self._model = None
    self._posterior_model = None

    # Load primary models
    self._encoder = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)
    for param in self._encoder.parameters():
      param.requires_grad = False
    self._tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    self._embed_size = self._encoder.transformer.config.hidden_size

    # Load encoded, tokenized sentences into data loader
    encoded_sentences = []
    for sent in self._base_dataset["full_sentence"]:
      encoded_sent = self._tokenizer.encode(sent)
      # TODO(eric): should we be decorating sentences with BOS/EOS?
      encoded_sent.insert(0, self._tokenizer.bos_token_id)
      encoded_sent.append(self._tokenizer.eos_token_id)
      encoded_sentences.append(
          torch.tensor(encoded_sent, dtype=torch.long, device=self._device))
    data_loader = torch.utils.data.DataLoader(
        dataset=Dataset(encoded_sentences,
                        self._base_dataset["encoded_intent_idxs"]),
        batch_size=64,
        collate_fn=collate_fn,
        shuffle=False)

    # Compute average reps
    self._base_dataset["cached_rep"] = []
    for batch_idx, (x, y) in enumerate(data_loader):
      with torch.no_grad():
        x = x.to(self._device)
        mask = x.ne(0).unsqueeze(2).repeat(1, 1, self._embed_size).float().to(
            self._device).detach()
        hidden, _ = self._encoder.transformer(x)
        masked_hidden = hidden * mask
        avg_rep = torch.sum(
            masked_hidden, dim=1) / (torch.sum(mask, dim=1).detach() + EPSILON)
        avg_rep = avg_rep.cpu().detach()
        avg_rep_list = torch.unbind(avg_rep.unsqueeze(1))
        self._base_dataset["cached_rep"] += avg_rep_list

    # Combine cached reps into single tensor
    self._base_dataset["cached_rep"] = torch.cat(
        self._base_dataset["cached_rep"]).unsqueeze(1)

  def _reset(self, batch_size=32, iterations=50):
    """Initialize selective sampling task."""

    # Create model
    self._model = ClassificationHead(
        class_size=self._intents_count,
        embed_size=self._embed_size).to(self._device)
    self._model.train()

    # Train inner-model on support set
    if self._seed_size:
      self._fit_model(batch_size, iterations)

    # Create posterior model
    self._posterior_model = ClassificationHead(
        class_size=self._intents_count,
        embed_size=self._embed_size).to(self._device)
    self._posterior_model.train()

    # Train posterior model
    posterior_labels = np.full((self._pool_size, self._intents_count),
                               fill_value=1,
                               dtype=np.int32)
    for i in range(self._pool_size):
      posterior_labels[
          i] = posterior_labels[i] & self._dataset["encoded_intent_idxs"][i]
    data_loader = torch.utils.data.DataLoader(
        dataset=Dataset(self._dataset["cached_rep"],
                        posterior_labels.astype(np.float32)),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=cached_collate_fn)
    self._fit_model(batch_size,
                    iterations,
                    data_loader=data_loader,
                    model=self._posterior_model)

  def _fit_model(self,
                 batch_size=32,
                 iterations=50,
                 data_loader=None,
                 model=None):
    """"Fit underlying learner"""

    self._model.train(True)
    for param in self._encoder.parameters():
      param.requires_grad = False

    # Load dataset and mask
    assert np.sum(self._support_indices[:self._used_labels] == -1) == 0
    if data_loader is None:
      indices = np.unique(self._support_indices[:self._used_labels])
      data_loader = torch.utils.data.DataLoader(
          dataset=Dataset(self._dataset["cached_rep"][indices],
                          self._pot_labels[indices].astype(np.float32)),
          batch_size=batch_size,
          shuffle=True,
          collate_fn=cached_collate_fn)
    atomic_mask = torch.tensor(self._atomic_mask,
                               dtype=torch.float).to(self._device)

    # Model training
    model = model or self._model
    optimizer = optim.Adam(model.parameters(),
                           lr=0.002)  # Reinitialize annealing each time.
    epochs = math.ceil(float(batch_size * iterations) / self._used_labels)
    for i in range(epochs):
      for batch_idx, (input_t, target_t) in enumerate(data_loader):
        input_t, target_t = input_t.to(self._device), target_t.to(
            self._device)
        optimizer.zero_grad()
        output_t = model(input_t)
        loss = atom_nll_loss(output_t, target_t, atomic_mask)
        loss.backward(retain_graph=True)
        optimizer.step()

  def eval(self):
    self._model.train(False)
    data_loader = torch.utils.data.DataLoader(
        dataset=Dataset(self._dataset["cached_rep"],
                        np.array([self.mask(i) for i in
                                  range(self._pool_size)]).astype(np.float32)),
        batch_size=self._batch_size,
        shuffle=False,
        collate_fn=cached_collate_fn)
    softmax_op = torch.nn.Softmax(dim=1)
    results = []
    for batch_idx, (x, y) in enumerate(data_loader):
      with torch.no_grad():
        x = x.to(self._device)
        logits = self._model(x)
        predicted = torch.argmax(softmax_op(logits) * y, dim=1)
        results += predicted.data.cpu().numpy().tolist()

    agreements = []
    for i, r in enumerate(results):
      agreements.append(r in self._dataset["intent_idxs"][i])
    return np.mean(agreements)

  def _predict(self):
    self._model.train(False)
    input_t = self._dataset["cached_rep"][self._current_idx]
    logits = self._model(input_t).data.cpu().numpy().flatten()
    return np.argmax(softmax(logits) * self.mask())

  def current_dist(self):
    self._model.train(False)
    softmax_op = torch.nn.Softmax(dim=1)
    input_t = self._dataset["cached_rep"][self._current_idx]
    output_t = softmax_op(
        self._model(input_t)).data.cpu().numpy().flatten()
    output_t = output_t * self.mask()
    return output_t / sum(output_t)

  def current_posterior_dist(self):
    self._model.train(False)
    softmax_op = torch.nn.Softmax(dim=1)
    input_t = self._dataset["cached_rep"][self._current_idx]
    output_t = softmax_op(
        self._posterior_model(input_t)).data.cpu().numpy().flatten()
    output_t = output_t * self.mask()
    return output_t / sum(output_t)

