import pickle

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config
import preprocessing as pp

def predict_test(model:str, test_data:pd.DataFrame= config.MODIFIED_TEST):

    # path to model
    model_path = f"{config.MODEL_DIR}/PRETRAIN_WORD2VEC_{model}/"

    # read data
    df_test = pd.read_csv(test_data)

    # do cleaning to text
    df_test[config.CLEANED_TEXT] = df_test[config.TEXT].apply(pp.clean_tweet)

    # loading tokenizer
    with open(f'{model_path}tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # convert tokens to sequences and pad them
    data_values = tokenizer.texts_to_sequences(df_test[config.CLEANED_TEXT].values)
    X_padded = pad_sequences(data_values, maxlen=config.MAXLEN)

    # load the classifier
    clf = load_model(f"{model_path}{model}_Word2Vec .h5")
    predictions = clf.predict_classes(X_padded, verbose=-1)

    return predictions

if __name__ == "__main__":
    submission = predict_test(model="LSTM")
    sample_sub = pd.read_csv(config.SUBMISSION)
    sample_sub.loc[:, config.TARGET] = submission
    sample_sub.to_csv(f"{config.MODEL_DIR}PRETRAIN_WORD2VEC_LSTM/LSTM.csv", index=False)