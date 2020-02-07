from ActiveDialogue.datasets.common.turn import Turn


class Dialogue:

    def __init__(self, dialogue_id, turns):
        self.id = dialogue_id
        self.turns = turns

    def __len__(self):
        return len(self.turns)

    def to_dict(self):
        return {
            'dialogue_id': self.id,
            'turns': [t.to_dict() for t in self.turns]
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d['dialogue_id'], [Turn.from_dict(t) for t in d['turns']])

    @classmethod
    def annotate_raw(cls, raw):
        return cls(raw['dialogue_idx'],
                   [Turn.annotate_raw(t) for t in raw['dialogue']])
