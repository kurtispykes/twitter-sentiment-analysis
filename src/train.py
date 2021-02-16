import os
import pickle

import pandas as pd
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
    df_train[config.CLEANED_TEXT] = df_train[config.TEXT].apply(lambda x: pp.clean_tweet(x))
    df_test[config.CLEANED_TEXT] = df_test[config.TEXT].apply(lambda x: pp.clean_tweet(x))

    # save the modified train and test data
    df_train.to_csv(config.MODIFIED_TRAIN, index=False)
    df_test.to_csv(config.MODIFIED_TEST, index=False)
    del df_test

    # convert text to numerical representation
    tokenizer = Tokenizer(oov_token="<unk>")
    tokenizer.fit_on_texts(df_train[config.CLEANED_TEXT])

    # path to save model
    model_path = f"{config.MODEL_DIR}/PRETRAIN_WORD2VEC_{model}/"

    # saving tokenizer
    with open(f'{model_path}tokenizer.pkl', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # pad the sequences
    X = pad_sequences(tokenizer.texts_to_matrix(df_train[config.CLEANED_TEXT]), maxlen=config.MAXLEN)

    # get the pretrained word embeddings and prepare embedding layer
    embedding_matrix = f.get_word2vec_enc(tokenizer.word_index.items(), config.PRETRAINED_WORD2VEC)
    embedding_layer = Embedding(input_dim=config.VOCAB_SIZE,
                                output_dim=config.EMBED_SIZE,
                                weights=[embedding_matrix],
                                input_length=config.MAXLEN,
                                trainable=False)

    # target values
    y = df_train[config.RELABELED_TARGET].values

    # initialize kfold
    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=False)
    predictions = None
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=X, y=y)):
        X_train, X_val = X[train_idx, :], X[val_idx, :]
        y_train, y_val = y[train_idx], y[val_idx]
        #
        # train the model
        clf = my_LSTM(embedding_layer)
        model_history = clf.fit(X_train, y_train,
                                epochs=config.N_EPOCHS,
                                verbose=1)

        # evaluate test
        evaluation = clf.evaluate(X_val, y_val)

        # print results
        print(f"Fold {fold}")
        print(f"Train Acc: {model_history.history['accuracy']}\n",
              f"Val scores: {list(zip(clf.metrics_names, evaluation))}\n")

        # checking the folder exist
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # persist the model
        clf.save(f"{model_path}/{model}_Word2Vec_{fold}.h5")

if __name__ == "__main__":
    run_training("LSTM")


