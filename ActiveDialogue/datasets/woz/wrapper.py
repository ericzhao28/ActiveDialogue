from ActiveDialogue.config import mnt_dir
from ActiveDialogue.datasets.common import Ontology, Vocab, Dataset

import logging
import json


def load_dataset(splits=('train', 'dev', 'test')):
    with open(os.path.join(mnt_dir, 'ontology.json')) as f:
        ontology = Ontology.from_dict(json.load(f))

    with open(os.path.join(mnt_dir, 'vocab.json')) as f:
        vocab = Vocab.from_dict(json.load(f))

    with open(os.path.join(mnt_dir, 'emb.json')) as f:
        E = json.load(f)

    dataset = {}
    for split in splits:
        with open(os.path.join(mnt_dir, '{}.json'.format(split))) as f:
            logging.warn('loading split {}'.format(split))
            dataset[split] = Dataset.from_dict(json.load(f))

    logging.info('dataset sizes: {}'.format(pformat({k: len(v) for k, v in dataset.items()})))
    return dataset, ontology, vocab, E
