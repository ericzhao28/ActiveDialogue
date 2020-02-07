from ActiveDialogue.datasets.common import annotate


class Turn:

    def __init__(self,
                 turn_id,
                 transcript,
                 turn_label,
                 belief_state,
                 system_acts,
                 system_transcript,
                 num=None):
        self.id = turn_id
        self.transcript = transcript
        self.turn_label = turn_label
        self.belief_state = belief_state
        self.system_acts = system_acts
        self.system_transcript = system_transcript
        self.num = num or {}

    def to_dict(self):
        return {
            'turn_id': self.id,
            'transcript': self.transcript,
            'turn_label': self.turn_label,
            'belief_state': self.belief_state,
            'system_acts': self.system_acts,
            'system_transcript': self.system_transcript,
            'num': self.num
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def annotate_raw(cls, raw):
        system_acts = []
        for a in raw['system_acts']:
            if isinstance(a, list):
                s, v = a
                system_acts.append(['inform'] + s.split() + ['='] + v.split())
            else:
                system_acts.append(['request'] + a.split())
        # NOTE: fix inconsistencies in data label
        fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number'}
        return cls(
            turn_id=raw['turn_idx'],
            transcript=annotate(raw['transcript']),
            system_acts=system_acts,
            turn_label=[[
                fix.get(s.strip(), s.strip()),
                fix.get(v.strip(), v.strip())
            ] for s, v in raw['turn_label']],
            belief_state=raw['belief_state'],
            system_transcript=raw['system_transcript'],
        )

    def numericalize_(self, vocab):
        self.num['transcript'] = vocab.word2index(
            ['<sos>'] + [w.lower() for w in self.transcript + ['<eos>']],
            train=True)
        self.num['system_acts'] = [
            vocab.word2index(['<sos>'] + [w.lower()
                                          for w in a] + ['<eos>'],
                             train=True)
            for a in self.system_acts + [['<sentinel>']]
        ]
