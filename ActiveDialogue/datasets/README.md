# Datasets module

## Installing and preprocessing data
Instructions here borrowed from [GLAD's repository](https://github.com/salesforce/glad).

This project uses Stanford CoreNLP to annotate the dataset.
In particular, we use the [Stanford NLP Stanza python interface](https://github.com/stanfordnlp/stanza).
To run the server, do

```
docker run --name corenlp -d -p 9000:9000 vzhong/corenlp-server
```

The first time you preprocess the data, we will [download word embeddings and character embeddings and put them into a SQLite database](https://github.com/vzhong/embeddings), which will be slow.
Subsequent runs will be much faster.

```
python -m ActiveDialogue.datasets.woz.preprocess
```

## Supported Datasets
Currently, the Woz restaurant corpus is supported. Multi-Woz support will be added soon.

## Structure
We employ four primary datastructures: Dataset, Ontology, Dialogue and Turn. The primary logic is found in Dataset.

Each dataset provides its own module with a `preprocess` script and a `wrapper` submodule. Only the `wrapper` is used in normal execution.
