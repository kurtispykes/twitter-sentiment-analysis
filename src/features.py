import gensim
import numpy as np

def fn_pre_process_data(tweet):
    """
    Generator function to convert a document to a list of tokens
    :param tweet:
    :return:
    """
    for rec in tweet:
        yield gensim.utils.simple_preprocess(rec)

def get_vector_representations(token_list, vector, k=150):
    """
    Get the word embedding vectors
    :param token_list: list of tokens from cleaned text
    :param vector: the word embedding vectors
    :param k: dimensions
    :return:
    """
    # an empty token list with be an empty k dimensional array
    if len(token_list) < 1:
        return np.zeros(k)
    else:
        # we get the word embedding if the word is in the vector
        vectorize = [vector[word] if word in vector else np.random.rand(k) for word in token_list]

    total = np.sum(vectorize, axis=0)
    return total/len(vectorize)

def get_embeddings(token_list, vector):
    """
    Store embeddings in list format
    :param token_list: list of tokens
    :param vector: word2vec embeddings
    :return: list of embeddings
    """
    embeddings = token_list.apply(lambda x: get_vector_representations(x, vector))
    return list(embeddings)