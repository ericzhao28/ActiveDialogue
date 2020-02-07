"""Global-Locally Self-Attentive DST architecture (1805.09655)"""

from ActiveDialogue.models.common import Model
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
import logging
import os
import re
import json
from collections import defaultdict
from pprint import pformat


class SelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, lens):
        batch_size, seq_len, d_feat = inp.size()
        inp = self.dropout(inp)
        scores = self.scorer(inp.contiguous().view(-1, d_feat)).view(
            batch_size, seq_len)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores.data[i, l:] = -np.inf
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
        return context


class GLADEncoder(nn.Module):
    """
    the GLAD encoder described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, din, dhid, slots, dropout=None):
        super().__init__()
        self.dropout = dropout or {}
        self.global_rnn = nn.LSTM(din,
                                  dhid,
                                  bidirectional=True,
                                  batch_first=True)
        self.global_selfattn = SelfAttention(2 * dhid,
                                             dropout=self.dropout.get(
                                                 'selfattn', 0.))
        for s in slots:
            setattr(
                self, '{}_rnn'.format(s),
                nn.LSTM(din,
                        dhid,
                        bidirectional=True,
                        batch_first=True,
                        dropout=self.dropout.get('rnn', 0.)))
            setattr(
                self, '{}_selfattn'.format(s),
                SelfAttention(2 * dhid,
                              dropout=self.dropout.get('selfattn', 0.)))
        self.slots = slots
        self.beta_raw = nn.Parameter(torch.Tensor(len(slots)))
        nn.init.uniform_(self.beta_raw, -0.01, 0.01)

    def beta(self, slot):
        return F.sigmoid(self.beta_raw[self.slots.index(slot)])

    def forward(self, x, x_len, slot, default_dropout=0.2):
        local_rnn = getattr(self, '{}_rnn'.format(slot))
        local_selfattn = getattr(self, '{}_selfattn'.format(slot))
        beta = self.beta(slot)
        local_h = run_rnn(local_rnn, x, x_len)
        global_h = run_rnn(self.global_rnn, x, x_len)
        h = F.dropout(local_h, self.dropout.get(
            'local', default_dropout), self.training) * beta + F.dropout(
                global_h, self.dropout.get('global', default_dropout),
                self.training) * (1 - beta)
        c = F.dropout(local_selfattn(h, x_len),
                      self.dropout.get('local', default_dropout),
                      self.training) * beta + F.dropout(
                          self.global_selfattn(h, x_len),
                          self.dropout.get('global', default_dropout),
                          self.training) * (1 - beta)
        return h, c


class GLAD(Model):
    """
    the GLAD model described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, args, ontology, vocab):
        super().__init__()

        self.utt_encoder = GLADEncoder(args.demb,
                                       args.dhid,
                                       self.ontology.slots,
                                       dropout=args.dropout)
        self.act_encoder = GLADEncoder(args.demb,
                                       args.dhid,
                                       self.ontology.slots,
                                       dropout=args.dropout)
        self.ont_encoder = GLADEncoder(args.demb,
                                       args.dhid,
                                       self.ontology.slots,
                                       dropout=args.dropout)
        self.utt_scorer = nn.Linear(2 * args.dhid, 1)
        self.score_weight = nn.Parameter(torch.Tensor([0.5]))

    def forward(self, batch, labels, mask=False, training=False):
        # convert to variables and look up embeddings
        eos = self.vocab.word2index('<eos>')
        utterance, utterance_len = pad([e.num['transcript'] for e in batch],
                                       self.emb_fixed,
                                       self.device,
                                       pad=eos)
        acts = [
            pad(e.num['system_acts'], self.emb_fixed, self.device, pad=eos)
            for e in batch
        ]
        ontology = {
            s: pad(v, self.emb_fixed, self.device, pad=eos)
            for s, v in self.ontology.num.items()
        }

        ys = {}
        for s in self.ontology.slots:
            # for each slot, compute the scores for each value
            H_utt, c_utt = self.utt_encoder(utterance, utterance_len, slot=s)
            _, C_acts = list(
                zip(*[
                    self.act_encoder(a, a_len, slot=s) for a, a_len in acts
                ]))
            _, C_vals = self.ont_encoder(ontology[s][0],
                                         ontology[s][1],
                                         slot=s)

            # compute the previous action score
            q_acts = []
            for i, C_act in enumerate(C_acts):
                q_act, _ = attend(C_act.unsqueeze(0),
                                  c_utt[i].unsqueeze(0),
                                  lens=[C_act.size(0)])
                q_acts.append(q_act)
            y_acts = torch.cat(q_acts, dim=0).mm(C_vals.transpose(0, 1))

            # compute the utterance score
            C_acts = torch.cat(C_acts)
            q_utts, _ = attend(
                torch.repeat_interleave(H_utt.unsqueeze(0), C_vals.size(0),
                                        0),
                torch.repeat_interleave(C_vals.unsqueeze(1), len(batch), 1),
                lens=utterance_len)
            y_utts = self.utt_scorer(q_utts.transpose(0, 1)).squeeze(2)

            # combine the scores
            ys[s] = F.sigmoid(y_utts + self.score_weight * y_acts)

        if training:
            labels = {
                s: torch.Tensor(m).to(self.device) for s, m in labels.items()
            }

            loss = 0
            for s in self.ontology.slots:
                if mask:
                    loss += F.binary_cross_entropy(
                        ys[s], labels[s], reduction=None).mul(
                            mask[s]) / torch.sum(mask[s], dim=1)**args.gamma
                else:
                    loss += F.binary_cross_entropy(ys[s], labels[s])
        else:
            loss = torch.Tensor([0]).to(self.device)
        return loss, {s: v.data.tolist() for s, v in ys.items()}
