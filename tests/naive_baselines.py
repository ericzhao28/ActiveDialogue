from ActiveDialogue.strategies import naive_baselines
import random
import numpy as np

for i in range(100):
    random.seed(i)
    np.random.seed(i)

    example_pred = {"hello": np.array([[3,4,5,6], [3,4,5,6], [3,4,5,6]]), "awef": np.array([[3,4,2,1,2,3,2,2,2], [3,4,2,1,2,3,2,2,2], [3,4,2,1,2,3,2,2,2]])}
    batch_size = 3
    labeled = np.zeros(batch_size,)
    x = naive_baselines.random_singlets(example_pred)
    for v in x.values():
        for j in range(len(v)):
            assert len(np.where(v[j] != -1)) <= 1
            if np.any(v[j] != -1):
                labeled[j] += 1
    assert np.all(labeled == np.ones_like(labeled))

    for v in naive_baselines.passive_baseline(example_pred).values():
        assert np.all(v == -1)
