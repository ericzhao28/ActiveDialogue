# Environments Module
We implement an environment for maintaining a model and support set for dialogue state tracking tests. The primary environment is `DSTEnv`.

## DSTEnv
The primary state of the environment is the turn index, the label budget, and the support set.
Given that the seeding of the environment is often-costly, the environment has separate functions: `train_seed` for training (and saving) over the seed support, while `load_seed` is intended for day-to-day use. Following the seeding, environment agents can observe the current batch of observations using `observe`, request labels from the batch using `label`, and progress the stream with `step`. At any point, you can fit on the model with `fit` or evaluate on the complete test dataset using `eval`.
