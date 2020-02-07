from ActiveDialogue.datasets.woz import wrapper
import numpy as np


datasets, ontology, vocab, E = wrapper.load_dataset()

dataset = datasets["train"]
all_idxs, seed_idxs, num_turns = dataset.get_turn_idxs(50, 2, "singlepass")
all_idxs, seed_idxs, num_turns = dataset.get_turn_idxs(50, 2, "uniform")
all_idxs, seed_idxs, num_turns = dataset.get_turn_idxs(50, 0, "singlepass")
all_idxs, seed_idxs, num_turns = dataset.get_turn_idxs(50, 0, "uniform")
all_idxs, seed_idxs, num_turns = dataset.get_turn_idxs(10000, 2, "singlepass")
all_idxs, seed_idxs, num_turns = dataset.get_turn_idxs(10000, 2, "uniform")

next(dataset.batch(32, idxs=np.array([3,4,5,6,10]), shuffle=False))
next(dataset.batch(32, idxs=np.array([3,4,5,6,10]), shuffle=True))

dataset.get_labels(idxs=np.array([3,4,5,6,10]))
