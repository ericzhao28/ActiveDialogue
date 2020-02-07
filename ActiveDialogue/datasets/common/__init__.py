from stanza.nlp.corenlp import CoreNLPClient

client = None


def annotate(sent):
    global client
    if client is None:
        client = CoreNLPClient(
            default_annotators='ssplit,tokenize'.split(','))
    words = []
    for sent in client.annotate(sent).sentences:
        for tok in sent:
            words.append(tok.word)
    return words
