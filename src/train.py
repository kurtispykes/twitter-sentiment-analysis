import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config
import preprocessing as pp
import features as f
import data_cleaning as data_clean
from lstm_model import my_LSTM

# GPU Use
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

def run_training(model:str) -> None:
    """
    Training our Machine Learning model and serializing to disc
    """
    # read train and test data
    df_train = pd.read_csv(config.ORIGINAL_TRAIN)
    df_test = pd.read_csv(config.TEST_DATA)

    # relabel mislabeled samples
    df_train = data_clean.relabel_target(df_train)

    # shuffle data
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    # clean the text
    df_train[config.CLEANED_TEXT] = df_train[config.TEXT].apply(pp.clean_tweet)
    df_test[config.CLEANED_TEXT] = df_test[config.TEXT].apply(pp.clean_tweet)

    # save the modified train and test data
    df_train.to_csv(config.MODIFIED_TRAIN, index=False)
    df_test.to_csv(config.MODIFIED_TEST, index=False)
    del df_test

    # convert text to numerical representation
    tokenizer = Tokenizer(oov_token="<unk>")
    tokenizer.fit_on_texts(df_train[config.CLEANED_TEXT])

    # path to save model
    model_path = f"{config.MODEL_DIR}/PRETRAIN_WORD2VEC_{model}/"

    # checking the folder exist
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # saving tokenizer
    with open(f'{model_path}tokenizer.pkl', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # pad the sequences
    X_padded = pad_sequences(tokenizer.texts_to_sequences(df_train[config.CLEANED_TEXT].values), maxlen=config.MAXLEN)

    # get the pretrained word embeddings and prepare embedding layer
    embedding_matrix = f.get_word2vec_enc(tokenizer.word_index.items(), config.PRETRAINED_WORD2VEC)
    embedding_layer = Embedding(input_dim=config.VOCAB_SIZE,
                                output_dim=config.EMBED_SIZE,
                                weights=[embedding_matrix],
                                input_length=config.MAXLEN,
                                trainable=False)

    # target values
    y = df_train[config.RELABELED_TARGET].values

    # train a single model
    clf = my_LSTM(embedding_layer)
    clf.fit(X_padded, y,
            epochs=config.N_EPOCHS,
            verbose=1)

    # persist the model
    clf.save(f"{model_path}/{model}_Word2Vec.h5")

if __name__ == "__main__":
    run_training("LSTM")


