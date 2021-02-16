import numpy as np
import gensim.downloader as api

import config

def get_word2vec_enc(corpus: list,  gensim_pretrained_emb:str) -> list:
    """
    Get the W2V value for each word withing
    :param text: The text we want to get embeddings for
    :param embed_size: Dimension output for pretrained embeddings
    :param pretrained_emb: The pretrained embedding to use
    :return: words encoded as vectors
    """
    word_vecs = api.load(gensim_pretrained_emb)
    embedding_weights = np.zeros((config.VOCAB_SIZE, config.EMBED_SIZE))
    for word, i in corpus:
        if word in word_vecs:
            embedding_weights[i] = word_vecs[word]
    return embedding_weights
