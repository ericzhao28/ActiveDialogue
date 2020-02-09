from ActiveDialogue.datasets.woz import wrapper
import numpy as np


datasets, ontology, vocab, E = wrapper.load_dataset()

dataset = datasets["train"]
all_ptrs, seed_ptrs, num_turns = dataset.get_turn_ptrs(50, 2, "singlepass")
all_ptrs, seed_ptrs, num_turns = dataset.get_turn_ptrs(50, 2, "uniform")
all_ptrs, seed_ptrs, num_turns = dataset.get_turn_ptrs(50, 0, "singlepass")
all_ptrs, seed_ptrs, num_turns = dataset.get_turn_ptrs(50, 0, "uniform")
all_ptrs, seed_ptrs, num_turns = dataset.get_turn_ptrs(10000, 2, "singlepass")
all_ptrs, seed_ptrs, num_turns = dataset.get_turn_ptrs(10000, 2, "uniform")

next(dataset.batch(32, ptrs=np.array([3,4,5,6,10]), shuffle=False))
next(dataset.batch(32, ptrs=np.array([3,4,5,6,10]), shuffle=True))

dataset.get_labels(ptrs=np.array([3,4,5,6,10]))
