# Download and annotate data

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

The raw data will be stored in `data/woz/raw` of the container.
The annotation results will be stored in `data/woz/ann` of the container.

# Contribution

Pull requests are welcome!
If you have any questions, please create an issue or contact the corresponding author at `victor <at> victorzhong <dot> com`.
