import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
import logging
import os
import pdb
import re
import json
from collections import defaultdict
from pprint import pformat


def pad(seqs, emb, device, pad=0):
    lens = [len(s) for s in seqs]
    max_len = max(lens)
    padded = torch.LongTensor(
        [s + (max_len - l) * [pad] for s, l in zip(seqs, lens)])
    return emb(padded.to(device)), lens


def run_rnn(rnn, inputs, lens):
    # sort by lens
    order = np.argsort(lens)[::-1].tolist()
    reindexed = inputs.index_select(0, inputs.data.new(order).long())
    reindexed_lens = [lens[i] for i in order]
    packed = nn.utils.rnn.pack_padded_sequence(reindexed,
                                               reindexed_lens,
                                               batch_first=True)
    outputs, _ = rnn(packed)
    padded, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                 batch_first=True,
                                                 padding_value=0.)
    reverse_order = np.argsort(order).tolist()
    recovered = padded.index_select(0, inputs.data.new(reverse_order).long())
    # reindexed_lens = [lens[i] for i in order]
    # recovered_lens = [reindexed_lens[i] for i in reverse_order]
    # assert recovered_lens == lens
    return recovered


def attend(seq, cond, lens):
    """
    attend over the sequences `seq` using the condition `cond`.
    """
    scores = cond.unsqueeze(-2).expand_as(seq).mul(seq).sum(-1)
    max_len = max(lens)
    for i, l in enumerate(lens):
        if l < max_len:
            if len(scores.shape) == 3:
                scores.data[:, i, l:] = -np.inf
            else:
                scores.data[i, l:] = -np.inf
    scores = F.softmax(scores, dim=-2)
    context = scores.unsqueeze(-1).expand_as(seq).mul(seq).sum(-2)
    return context, scores


class FixedEmbedding(nn.Embedding):
    """
    this is the same as `nn.Embedding` but detaches the result from the graph and has dropout after lookup.
    """

    def __init__(self, *args, dropout=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = dropout

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        out.detach_()
        return F.dropout(out, self.dropout, self.training)


class Model(nn.Module):

    def __init__(self, args, ontology, vocab, dout=None):
        super().__init__()
        self.optimizer = None
        self.args = args
        if dout is None:
            self.dout = args.dout
        else:
            self.dout = dout
        self.vocab = vocab
        self.ontology = ontology
        self.emb_fixed = FixedEmbedding(len(vocab),
                                        args.demb,
                                        dropout=args.dropout.get('emb', 0.2))

    @property
    def device(self):
        if self.args.device is not None and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.args.lr)

    def load_emb(self, Eword):
        new = self.emb_fixed.weight.data.new
        self.emb_fixed.weight.data.copy_(new(Eword))

    def get_train_logger(self):
        logger = logging.getLogger('train-{}'.format(self.__class__.__name__))
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s'
        )
        file_handler = logging.FileHandler(
            os.path.join(self.dout, 'train.log'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def extract_predictions(self, scores, threshold=0.5):
        batch_size = len(list(scores.values())[0])
        predictions = [set() for i in range(batch_size)]
        for s in self.ontology.slots:
            for i, p in enumerate(scores[s]):
                triggered = [(s, v, p_v)
                             for v, p_v in zip(self.ontology.values[s], p)
                             if p_v > threshold]
                if s == 'request':
                    # we can have multiple requests predictions
                    predictions[i] |= set([(s, v) for s, v, p_v in triggered])
                elif triggered:
                    # only extract the top inform prediction
                    sort = sorted(triggered,
                                  key=lambda tup: tup[-1],
                                  reverse=True)
                    predictions[i].add((sort[0][0], sort[0][1]))
        return predictions

    def run_pred(self, dev, args):
        self.eval()
        predictions = []
        for batch, batch_labels in dev.batch(batch_size=args.batch_size):
            loss, scores = self.forward(batch, batch_labels)
            predictions += self.extract_predictions(scores)
        return predictions

    def run_eval(self, dev, args):
        predictions = self.run_pred(dev, args)
        return dev.evaluate_preds(predictions)

    def save_config(self):
        fname = '{}/config.json'.format(self.dout)
        with open(fname, 'wt') as f:
            logging.info('saving config to {}'.format(fname))
            json.dump(vars(self.args), f, indent=2)

    @classmethod
    def load_config(cls, fname, ontology, **kwargs):
        with open(fname) as f:
            logging.info('loading config from {}'.format(fname))
            args = object()
            for k, v in json.load(f):
                setattr(args, k, kwargs.get(k, v))
        return cls(args, ontology)

    def save(self, summary, identifier):
        fname = '{}/{}.t7'.format(self.dout, identifier)
        logging.info('saving model to {}'.format(fname))
        state = {
            'args': vars(self.args),
            'model': self.state_dict(),
            'summary': summary,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, fname)

    def load_id(self, identifier):
        fname = '{}/{}.t7'.format(self.dout, identifier)
        try:
            self.load(fname)
            return True
        except FileNotFoundError:
            return False

    def load(self, fname):
        logging.info('loading model from {}'.format(fname))
        state = torch.load(fname)
        self.load_state_dict(state['model'])
        self.set_optimizer()
        self.optimizer.load_state_dict(state['optimizer'])

    def get_saves(self, directory=None):
        if directory is None:
            directory = self.dout
        files = [f for f in os.listdir(directory) if f.endswith('.t7')]
        scores = []
        for fname in files:
            re_str = r'dev_{}=([0-9\.]+)'.format(self.args.stop)
            dev_acc = re.findall(re_str, fname)
            if dev_acc:
                score = float(dev_acc[0].strip('.'))
                scores.append((score, os.path.join(directory, fname)))
        if not scores:
            raise Exception('No files found!')
        scores.sort(key=lambda tup: tup[0], reverse=True)
        return scores

    def prune_saves(self, n_keep=5):
        scores_and_files = self.get_saves()
        if len(scores_and_files) > n_keep:
            for score, fname in scores_and_files[n_keep:]:
                os.remove(fname)

    def load_best_save(self, directory):
        if directory is None:
            directory = self.dout

        scores_and_files = self.get_saves(directory=directory)
        if scores_and_files:
            assert scores_and_files, 'no saves exist at {}'.format(directory)
            score, fname = scores_and_files[0]
            self.load(fname)

    def forward(self, batch, labels=None, training=False):
        ys = self.infer(batch)
        if training:
            keys = list(labels.keys())
            flatlabels = torch.Tensor(np.concatenate([labels[k] for k in keys]), axis=1).to(self.device)
            flatys = torch.cat([ys[k] for k in keys], dim=1)
            loss = F.binary_cross_entropy(flatys, flatlabels)
        else:
            loss = torch.Tensor([0]).to(self.device)
        return loss, {s: v.data.tolist() for s, v in ys.items()}

    def partial_forward(self, batch, labels=None, training=False, mask=None):
        ys = self.infer(batch)
        if training:
            keys = list(labels.keys())
            flatlabels = torch.Tensor(np.concatenate([labels[k] for k in keys], axis=1)).to(self.device)
            flatys = torch.cat([ys[k] for k in keys], dim=1)
            flatmask = torch.Tensor(np.concatenate([mask[k] for k in keys], axis=1)).to(self.device)
            loss = torch.mean(F.binary_cross_entropy(flatys, flatlabels, reduce='none').mul(flatmask))
        else:
            loss = torch.Tensor([0]).to(self.device)
        return loss, {s: v.data.tolist() for s, v in ys.items()}

    def bag_forward(self,
                    batch,
                    bag,
                    feedback,
                    training=False,
                    sl_reduction=False,
                    optimistic_weighting=False):
        ys = self.infer(batch)
        feedback = torch.tensor(feedback).float().to(self.device)

        if training:
            weight = None
            for s in self.ontology.slots:
                tbag = torch.zeros_like(ys[s])
                tbag_idxs = np.array([(ii, j) for ii, i in enumerate(bag[s]) for j in i])
                tbag[tbag_idxs[:, 0], tbag_idxs[:, 1]] = 1
                bag[s] = tbag.to(self.device)

                if weight is None:
                    weight = torch.sum(bag[s], dim=1)
                else:
                    weight += torch.sum(bag[s], dim=1)
            if not optimistic_weighting:
                weight[feedback == 0] = 1

            loss = 0
            flat_ys = []
            flat_bag = []
            for s in self.ontology.slots:
                flat_ys.append(ys[s])
                flat_bag.append(bag[s])
            flat_ys = torch.cat(flat_ys, dim=1)
            flat_bag = torch.cat(flat_bag, dim=1)

            if sl_reduction:
                loss = F.binary_cross_entropy(flat_ys, feedback.unsqueeze(1).expand_as(flat_ys), reduction='none')
                loss = torch.sum(loss.mul(flat_bag))
            else:
                loss = feedback * torch.sum(torch.log(flat_ys + 1e-8).mul(flat_bag))
                flat_ys = torch.pow(flat_ys, flat_bag)
                loss += (1 - feedback) * torch.log(1 - torch.prod(flat_ys) + 1e-8)
            loss = torch.sum(loss / weight.unsqueeze(1))
        else:
            loss = torch.Tensor([0]).to(self.device)

        return loss, {s: v.data.tolist() for s, v in ys.items()}
