from ActiveDialogue.config import mnt_dir
from ActiveDialogue.datasets.common.ontology import Ontology
from ActiveDialogue.datasets.common.dataset import Dataset
from vocab import Vocab

import logging
import json
import os


def load_dataset(splits=('train', 'dev', 'test')):
    with open(os.path.join(mnt_dir + "/woz/ann", 'ontology.json')) as f:
        ontology = Ontology.from_dict(json.load(f))

    with open(os.path.join(mnt_dir + "/woz/ann", 'vocab.json')) as f:
        vocab = Vocab.from_dict(json.load(f))

    with open(os.path.join(mnt_dir + "/woz/ann", 'emb.json')) as f:
        E = json.load(f)

    dataset = {}
    for split in splits:
        with open(os.path.join(mnt_dir + "/woz/ann",
                               '{}.json'.format(split))) as f:
            logging.warn('loading split {}'.format(split))
            dataset[split] = Dataset.from_dict(json.load(f), ontology)

    logging.info('dataset sizes: {}'.format(
        str({k: len(v) for k, v in dataset.items()})))
    return dataset, ontology, vocab, E
