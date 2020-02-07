# ActiveDialogue Environment

We implement an environment for maintaining a model and support set for dialogue state tracking tests. The primary environment is `DSTEnv`.

## DSTEnv
### Initialization
The environment seeds all library RNGs to ensure consistent seed set production (so we can re-use saves of our underlying models). The environment retains the caller's args.

The dataset is initialized,

