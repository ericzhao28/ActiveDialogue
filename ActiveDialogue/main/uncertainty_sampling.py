import numpy as np
import random

random.seed(0)
np.random.seed(0)

from ActiveDialogue.environment.linear_gpt2_classification_env import \
    LinearGPT2ClassificationEnv
from ActiveDialogue.datasets.snips import wrapper as snips_wrapper
from ActiveDialogue.datasets.utils import gpt2_retokenization
from ActiveDialogue.strategies import classification_baselines
from transformers import GPT2Tokenizer
import torch


def run_baseline(title, strategy, env, **kargs):
  torch.cuda.empty_cache()
  score_prior = []
  score_post = []
  next_actions = []
  for i in range(5):
    reward, evaluation, ended = env.reset("singlepass",
                                          pool_size=20,
                                          budget=20,
                                          seed_size=10,
                                          iterations=20,
                                          batch_size=5,
                                          question_cap=5,
                                          fit_period=5)
    score_prior.append(env.eval())
    while not ended:
      if not next_actions:
        next_actions = strategy(env, **kargs)
      action = next_actions.pop(0)
      reward, evaluation, ended = env.step(action)
    score_post.append(evaluation)
  print("{} baseline: {} - {} = {}".format(
      title, np.mean(score_post), np.mean(score_prior),
      np.mean(score_post) - np.mean(score_prior)))


# Main environment
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
env = LinearGPT2ClassificationEnv(
    snips_wrapper, tokenizer, gpt2_retokenization, test=False, precap=200)
print("Env finished loading.")

# Random baseline
run_baseline("Random", classification_baselines.random_baseline, env)

# Passive baseline
run_baseline("Passive", classification_baselines.passive_baseline, env)

# Proactive optimal baseline
run_baseline("Proactive optimal",
             classification_baselines.proactive_optimal_baseline, env)

# Proactive optimal baseline
run_baseline("Proactive",
             classification_baselines.proactive_baseline,
             env,
             posterior=False)

# Posterior proactive optimal baseline
run_baseline("Posterior proactive",
             classification_baselines.proactive_baseline,
             env,
             posterior=True)

# Entropy threshold
run_baseline("Entropy",
             classification_baselines.entropy_threshold,
             env,
             threshold=0.1,
             posterior=False)

# Posterior-entropy threshold
run_baseline("Posterior entropy",
             classification_baselines.entropy_threshold,
             env,
             threshold=0.1,
             posterior=True)

# ERC threshold
run_baseline("ERC",
             classification_baselines.erc_threshold,
             env,
             threshold=5,
             posterior=False)

# Posterior-ERC threshold
run_baseline("Posterior ERC",
             classification_baselines.erc_threshold,
             env,
             threshold=4,
             posterior=True)

# EDC threshold
run_baseline("EDC",
             classification_baselines.edc_threshold,
             env,
             threshold=1.5,
             posterior=False)

# Posterior-EDC threshold
run_baseline("Posterior EDC",
             classification_baselines.edc_threshold,
             env,
             threshold=2,
             posterior=True)

