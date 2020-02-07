from ActiveDialogue.datasets.common.dialogue import Dialogue
from ActiveDialogue.datasets.common.ontology import Ontology
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm


class Dataset:

    def __init__(self, dialogues, ontology, shuffle_dlgs=False):
        self.dialogues = dialogues

        dlg_idxs = np.arange(len(dialogues))
        if shuffle_dlgs:
            dlg_idxs = np.random.permutation(dlg_idxs)

        self.turns = []
        self.turns_dlg = []
        for i, d in enumerate(self.dialogues[dlg_idxs]):
            for t in d:
                self.turns.append(t)
                self.turns_dlg.append(i)
        self.turns = np.array(self.turns, dtype=np.object_)
        self.turns_dlg = np.array(self.turns_dlg, dtype=np.int32)
        self.labels = {
            s: np.zeros(len(ontology.values[s]))
            for i in range(len(self.turns)) for s in ontology.slots
        }
        for i, e in enumerate(self.turns):
            for s, v in e.turn_label:
                self.labels[s][i][ontology.values[s].index(v)] = 1

    def get_turn_indices(self, pool_size, seed_size, sample_mode):
        seed_turn_idxs = np.where(self.turns_dlg < seed_size)
        train_turn_idxs = np.where(self.turns_dlg > seed_size)

        if sample_mode == "singlepass":
            selected_idxs = []
            while len(selected_idxs) < pool_size:
                selected_idxs.append(np.random.permutation(train_turn_idxs))
            selected_idxs = np.concatenate(selected_idxs)[:pool_size]
        elif sample_mode == "uniform":
            selected_idxs = np.random.choice(train_turn_idxs, pool_size)
        else:
            raise ValueError("ClassificationEnv: Invalid sample mode")
        selected_idxs = np.concatenate((seed_turn_idxs, selected_idxs))

        return selected_idxs, seed_turn_idxs, len(self.turns)

    def __len__(self):
        return len(self.dialogues)

    def to_dict(self):
        return {'dialogues': [d.to_dict() for d in self.dialogues]}

    @classmethod
    def from_dict(cls, d, ontology):
        return cls([Dialogue.from_dict(dd) for dd in d['dialogues']],
                   ontology)

    @classmethod
    def annotate_raw(cls, fname):
        with open(fname) as f:
            data = json.load(f)
            return cls([Dialogue.annotate_raw(d) for d in tqdm(data)])

    def numericalize_(self, vocab):
        for t in self.turns:
            t.numericalize_(vocab)

    def extract_ontology(self):
        slots = set()
        values = defaultdict(set)
        for t in self.turns:
            for s, v in t.turn_label:
                slots.add(s.lower())
                values[s].add(v.lower())
        return Ontology(sorted(list(slots)),
                        {k: sorted(list(v)) for k, v in values.items()})

    def batch(self, batch_size, indices=None, labels=None, shuffle=False):
        if indices is None:
            indices = np.arange(len(self.turns))
        if shuffle:
            indices = np.random.permutation(indices)

        turns = self.turns[indices]

        if labels is None:
            labels = self.labels[indices]
        else:
            labels = labels[indices]

        for i in tqdm(range(0, len(turns), batch_size)):
            yield turns[i:i + batch_size], labels[i:i + batch_size]

        if len(turns) % batch_size > 0:
            yield turns[-len(turns) % batch_size:], labels[-len(turns) %
                                                           batch_size:]

    def get_labels(self, indices):
        return self.labels[indices]

    def evaluate_preds(self, preds):
        request = []
        inform = []
        joint_goal = []
        fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number'}
        i = 0
        for d in self.dialogues:
            pred_state = {}
            for t in d.turns:
                gold_request = set([
                    (s, v) for s, v in t.turn_label if s == 'request'
                ])
                gold_inform = set([
                    (s, v) for s, v in t.turn_label if s != 'request'
                ])
                pred_request = set([
                    (s, v) for s, v in preds[i] if s == 'request'
                ])
                pred_inform = set([
                    (s, v) for s, v in preds[i] if s != 'request'
                ])
                request.append(gold_request == pred_request)
                inform.append(gold_inform == pred_inform)

                gold_recovered = set()
                pred_recovered = set()
                for s, v in pred_inform:
                    pred_state[s] = v
                for b in t.belief_state:
                    for s, v in b['slots']:
                        if b['act'] != 'request':
                            gold_recovered.add(
                                (b['act'], fix.get(s.strip(), s.strip()),
                                 fix.get(v.strip(), v.strip())))
                for s, v in pred_state.items():
                    pred_recovered.add(('inform', s, v))
                joint_goal.append(gold_recovered == pred_recovered)
                i += 1
        return {
            'turn_inform': np.mean(inform),
            'turn_request': np.mean(request),
            'joint_goal': np.mean(joint_goal)
        }

    def record_preds(self, preds, to_file):
        data = self.to_dict()
        i = 0
        for d in data['dialogues']:
            for t in d['turns']:
                t['pred'] = sorted(list(preds[i]))
                i += 1
        with open(to_file, 'wt') as f:
            json.dump(data, f)
