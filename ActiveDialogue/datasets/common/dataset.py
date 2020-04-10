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
        # labels and dialogue ptrs.
        self.turns = []  # Turn objects
        self.turns_dlg = []  # dialogues corresponding to each turn
        for i, dlg in enumerate(dialogues):
            for turn in dlg.turns:
                self.turns.append(turn)
                self.turns_dlg.append(i)
        self.turns = np.array(self.turns, dtype=np.object_)
        self.turns_dlg = np.array(self.turns_dlg, dtype=np.int32)
        logging.debug("Loaded {} turns.".format(len(self.turns)))
        logging.debug("Loaded {} dialogues.".format(len(self.dialogues)))
        self.turns_labels = {
            s: np.zeros((len(self.turns), len(ontology.values[s])),
                        dtype=np.float32) for s in ontology.slots
        }
        logging.debug("{} classes".format(sum([len(ontology.values[s]) for s in ontology.slots])))
        for i, turn in enumerate(self.turns):
            for slot, value in turn.turn_label:
                self.turns_labels[slot][
                    i, ontology.values[slot].index(value)] = 1

    def add_noise(self, fn, fp):
        if fp != 0:
            for s, v in self.turns_labels.items():
                v_mask = np.zeros_like(v, dtype=np.int32)
                v_mask[np.random.uniform(size=v.shape) < fp] = 1
                self.turns_labels[s][v_mask] = 1
        if fn != 0:
            for s, v in self.turns_labels.items():
                v_mask = np.zeros_like(v, dtype=np.int32)
                v_mask[np.random.uniform(size=v.shape) < fn] = 1
                self.turns_labels[s][v_mask] = 0

    def get_turn_ptrs(self, num_passes, seed_size, sample_mode):
        """Get a list of ptrs into Dataset.turns by SS env.
        Args:
            num_passes: Num of online passes.
            seed_size: Num of turns to reserve for seed turn ptrs.
            sample_mode: `singlepass` or `uniform`. Should ptrs be sampled
                         with or without replacement?
        Returns:
            all_ptrs: Turn ptrs for training.
            all_seed_ptrs: Non-repeating turn ptrs reserved for seeding.
            turn_ptr_cap: 1 + maximum turn ptr (number of turns from all
                          dialogues).
        """
        if seed_size >= len(self.turns):
            raise ValueError()

        # List of turn ptrs divided by dialogue index (first seed_size
        # are reserved for seeding).
        seed_ptrs = np.arange(seed_size)
        orig_nonseed_ptrs = np.arange(seed_size, len(self.turns))
        assert len(orig_nonseed_ptrs) > 0

        if sample_mode == "singlepass":
            nonseed_ptrs = np.tile(orig_nonseed_ptrs, num_passes)
        elif sample_mode == "uniform":
            nonseed_ptrs = np.random.choice(
                orig_nonseed_ptrs, num_passes * (len(self.turns) - seed_size))
        else:
            raise ValueError("ClassificationEnv: Invalid sample mode")

        # Combine seed and training ptrs (first len(seed_ptrs) are
        # seeded).
        # all_ptrs = np.concatenate((seed_ptrs, nonseed_ptrs))

        return nonseed_ptrs, seed_ptrs, len(self.turns)

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

    def batch(self,
              batch_size,
              ptrs=None,
              shuffle=False,
              return_ptrs=False,
              loop=False):
        """Grab a batch of dialogue turns and labels
        Args:
            batch_size: batch size.
            ptrs: specify which turn ptrs are to be included. By default,
                  all turns are included.
            shuffle: should the ordering of turns be shuffled?
        """
        while True:
            # Build array of relevant turn instances
            if ptrs is None:
                ptrs = np.arange(len(self.turns))
            if shuffle:
                ptrs = np.random.permutation(ptrs)
            turns = self.turns[ptrs]

            # Build array of labels
            labels = {s: v[ptrs] for s, v in self.turns_labels.items()}

            # Yield from our list of turns
            for i in range(0, len(turns), batch_size):
                if return_ptrs:
                    yield turns[i:i + batch_size], {
                        s: v[i:i + batch_size] for s, v in labels.items()
                    }, ptrs[i:i + batch_size]
                else:
                    yield turns[i:i + batch_size], {
                        s: v[i:i + batch_size] for s, v in labels.items()
                    }

            if not loop:
                break

    def get_labels(self, ptrs=None):
        """Return requested labels
        Args:
            ptrs: numpy array of turn ptrs
        Returns:
            labels: turn labels (ground truth)
        """
        if ptrs is None:
            return self.turns_labels
        return {
            slot: values[ptrs] for slot, values in self.turns_labels.items()
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
