import numpy as np
import random

random.seed(0)
np.random.seed(0)

from ActiveDialogue.environment import classification_env
from ActiveDialogue.datasets.atis import wrapper

env = classification_env.ClassificationEnv(wrapper, test=False, precap=50)

env.reset("singlepass", pool_size=50, budget=25, seed_size=5, batch_size=4,
          iterations=20, fit_period=1)
while True:
  env.display()
  action = input("Your action: ")
  try:
    action = int(action)
  except ValueError:
    pass
  env.step(action)
