from ActiveDialogue.datasets.common.dialogue import Dialogue
from ActiveDialogue.datasets.common.ontology import Ontology
import json
from collections import defaultdict
import pdb
import logging
import numpy as np
from tqdm import tqdm


class Dataset:

    def __init__(self, dialogues, ontology, shuffle_dlgs=False):
        """Initialize Dataset class

        Args:
            dialogues: List of Dialogue instances.
            ontology: Ontology instance.
            shuffle_dlgs: Should Dialogue instances be shuffled?
        """
        # Build optionally shuffled list of dialogues.
        dialogues = np.array(dialogues, dtype=np.object_)
        if shuffle_dlgs:
            dialogues = dialogues[np.random.permutation(
                np.arange(len(dialogues)))]
        self.dialogues = dialogues  # Dialogue objects

        # Build optionally shuffled list of turns and their corresponding
        # labels and dialogue idxs.
        self.turns = []  # Turn objects
        self.turns_dlg = []  # dialogues corresponding to each turn
        for i, dlg in enumerate(dialogues):
            for turn in dlg.turns:
                self.turns.append(turn)
                self.turns_dlg.append(i)
        self.turns = np.array(self.turns, dtype=np.object_)
        self.turns_dlg = np.array(self.turns_dlg, dtype=np.int32)
        print("Loaded {} turns.".format(len(self.turns)))
        print("Loaded {} dialogues.".format(len(self.dialogues)))
        self.turns_labels = {
            s: np.zeros((len(self.turns), len(ontology.values[s])),
                        dtype=np.float32) for s in ontology.slots
        }
        for i, turn in enumerate(self.turns):
            for slot, value in turn.turn_label:
                self.turns_labels[slot][i][ontology.values[slot].index(
                    value)] = 1

    def get_turn_idxs(self, pool_size, seed_size, sample_mode):
        """Get a list of idxs into Dataset.turns by SS env.
        Args:
            pool_size: Num of (non-seed) idxs requested.
            seed_size: Num of dialogues to reserve for seed turn idxs.
            sample_mode: `singlepass` or `uniform`. Should idxs be sampled
                         with or without replacement?
        Returns:
            all_idxs: Turn idxs for training.
            all_seed_idxs: Non-repeating turn idxs reserved for seeding.
            turn_idx_cap: 1 + maximum turn idx (number of turns from all
                          dialogues).
        """
        if seed_size >= len(self.dialogues):
            raise ValueError()

        # List of turn idxs divided by dialogue index (first seed_size
        # are reserved for seeding).
        seed_idxs = np.where(self.turns_dlg < seed_size)[0]
        orig_nonseed_idxs = np.where(self.turns_dlg >= seed_size)[0]

        if sample_mode == "singlepass":
            # Grab permutations of nonseed_idxs until pool_size is hit.
            nonseed_idxs = []
            for i in range(0, pool_size + len(orig_nonseed_idxs), len(orig_nonseed_idxs)):
                nonseed_idxs.append(np.random.permutation(orig_nonseed_idxs))
            nonseed_idxs = np.concatenate(nonseed_idxs)[:pool_size]
        elif sample_mode == "uniform":
            # Sample pool_size of nonseed_idxs with replacement.
            nonseed_idxs = np.random.choice(orig_nonseed_idxs, pool_size)
        else:
            raise ValueError("ClassificationEnv: Invalid sample mode")

        # Combine seed and training idxs (first len(seed_idxs) are
        # seeded).
        all_idxs = np.concatenate((seed_idxs, nonseed_idxs))

        return all_idxs, seed_idxs, len(self.turns)

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

    def batch(self, batch_size, idxs=None, labels=None, shuffle=False):
        """Grab a batch of dialogue turns and labels
        Args:
            batch_size: batch size.
            idxs: specify which turn idxs are to be included. By default,
                  all turns are included.
            labels: provide competing labels (1:1 with self.turns_labels)
            shuffle: should the ordering of turns be shuffled?
        """
        # Build array of relevant turn instances
        if idxs is None:
            idxs = np.arange(len(self.turns))
        if shuffle:
            idxs = np.random.permutation(idxs)
        turns = self.turns[idxs]

        # Build array of labels
        if labels is None:
            labels = {s: v[idxs] for s, v in self.turns_labels.items()}
        else:
            labels = {s: v[idxs] for s, v in labels.items()}

        # Yield from our list of turns
        for i in tqdm(range(0, len(turns), batch_size)):
            yield turns[i:i + batch_size], {
                s: v[i:i + batch_size] for s, v in labels.items()
            }

        # Handle leftovers
        leftover = len(turns) % batch_size
        if leftover > 0:
            yield turns[-leftover:], {
                s: v[-leftover:] for s, v in labels.items()
            }

    def get_labels(self, idxs=None):
        """Return requested labels
        Args:
            idxs: numpy array of turn idxs
        Returns:
            labels: turn labels (ground truth)
        """
        if idxs is None:
            return self.turns_labels
        return {
            slot: values[idxs] for slot, values in self.turns_labels.items()
        }

    def evaluate_preds(self, preds):
        request = []
        inform = []
        joint_goal = []
        fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number'}
        i = 0
        dialogues = self.dialogues
        for d in dialogues:
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
