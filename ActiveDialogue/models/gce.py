"""Globally-Conditioned Encoder DST architecture (1812.00899)"""

from ActiveDialogue.models.common import Model, run_rnn, pad, attend
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class SelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, cond, lens):
        batch_size, seq_len, d_feat = cond.size()
        cond = self.dropout(cond)
        scores = self.scorer(cond.contiguous().view(-1, d_feat)).view(
            batch_size, seq_len)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores.data[i, l:] = -np.inf
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
        return context


class GCEEncoder(nn.Module):
    """
    the GCE encoder described in https://arxiv.org/abs/1812.00899.
    """

    def __init__(self, din, dhid, dropout=None):
        super().__init__()
        self.dropout = dropout or {}
        self.global_rnn = nn.LSTM(2 * din,
                                  dhid,
                                  bidirectional=True,
                                  batch_first=True)
        self.global_selfattn = SelfAttention(din + 2 * dhid,
                                             dropout=self.dropout.get(
                                                 'selfattn', 0.))

    def forward(self, x, x_len, slot_emb, default_dropout=0.2):
        # I removed beta here... don't see why its needed

        x = torch.cat((slot_emb.unsqueeze(0).expand_as(x), x), dim=2)
        global_h = run_rnn(self.global_rnn, x, x_len)

        h = F.dropout(global_h, self.dropout.get('global', default_dropout),
                      self.training)

        hs = torch.cat((slot_emb.unsqueeze(0).expand_as(h), h), dim=2)
        c = F.dropout(self.global_selfattn(h, hs, x_len),
                      self.dropout.get('global', default_dropout),
                      self.training)
        return h, c


class GCE(Model):
    """
    the GCE model described in https://arxiv.org/abs/1812.00899.
    """

    def __init__(self, args, ontology, vocab):
        super().__init__(args, ontology, vocab)

        self.utt_encoder = GCEEncoder(args.demb,
                                      args.dhid,
                                      dropout=args.dropout)
        self.act_encoder = GCEEncoder(args.demb,
                                      args.dhid,
                                      dropout=args.dropout)
        self.ont_encoder = GCEEncoder(args.demb,
                                      args.dhid,
                                      dropout=args.dropout)
        self.utt_scorer = nn.Linear(2 * args.dhid, 1)
        self.score_weight = nn.Parameter(torch.Tensor([0.5]))
        self.args = args

    def infer(self, batch):
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
            # Add slot embedding
            s_emb = self.emb_fixed(
                torch.LongTensor([self.vocab.word2index(s.split()[0])
                                  ]).to(self.device))

            # for each slot, compute the scores for each value
            H_utt, c_utt = self.utt_encoder(utterance,
                                            utterance_len,
                                            slot_emb=s_emb)
            _, C_acts = list(
                zip(*[
                    self.act_encoder(a, a_len, slot_emb=s_emb)
                    for a, a_len in acts
                ]))
            _, C_vals = self.ont_encoder(ontology[s][0],
                                         ontology[s][1],
                                         slot_emb=s_emb)

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
            ys[s] = torch.sigmoid(y_utts + self.score_weight * y_acts)

        return ys
