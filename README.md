# Leveraging Error Handling Strategies for Active Learning

This repository contains experiments for applying active learning strategies for dialogue error handling. Specifically, we extend uncertainty sampling and error reduction methods to the partial multi instance learning and selective sampling settings.
Our most recent write-up can be found here on [Overleaf](https://www.overleaf.com/project/5e34166fd4449b00016cd549).

We build on Salesforce's GLAD model for dialogue state-tracking ([repo](https://github.com/salesforce/glad)), including GCE edits ([repo](https://github.com/elnaaz/GCE-Model)).

## Installation and usage
Install Python3.6+ and pip install our `requirements.txt` file. Run experiments as: `python3.6 -m ActiveDialogue.main.noop`.

## Repository Layout
The primary source-code can be found under `ActiveDialogue`. The modules are as listed:
* `datasets`: submodules for interfacing with datasets.
* `environments`: environment classes for selective sampling simulation.
* `main`: primary scripts for experiment replication.
* `models`: dialogue-state-tracking PyTorch models.
* `strategies`: baseline and proposed sampling strategy implementations.

Datasets are mounted under `mnt`; if missing, please contact the Authors (the files may have been too large to upload).
Unit test scripts are found in `tests`, although most require manual usage.
