import pickle

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config
import preprocessing as pp

def predict(model: str, test_data:pd.DataFrame = config.MODIFIED_TEST):

    # path to model
    model_path = f"{config.MODEL_DIR}/PRETRAIN_WORD2VEC_{model}/"

    # read data
    df_test = pd.read_csv(test_data)

    # do cleaning to text
    df_test[config.CLEANED_TEXT] = df_test[config.TEXT].apply(lambda x: pp.clean_tweet(x))

    # loading tokenizer
    with open(f'{model_path}tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # convert tokens to sequences and pad them
    data_values = tokenizer.texts_to_sequences(df_test[config.CLEANED_TEXT])
    X = pad_sequences(data_values, maxlen=config.MAXLEN)

    # use all 5 models to make predictions
    # divide the output of each model by 5
    predictions = None
    # loop through all folds
    for FOLD in range(5):
        # load the classifier
        clf = load_model(f"{model_path}{model}_Word2Vec_{FOLD}.h5")
        preds = np.argmax(clf.predict(X), axis=-1)

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    return predictions

if __name__ == "__main__":
    submission = predict(model="LSTM")
    sample_sub = pd.read_csv(config.SUBMISSION)
    sample_sub.loc[:, config.TARGET] = submission
    sample_sub.to_csv(f"{config.MODEL_DIR}PRETRAIN_WORD2VEC_LSTM/LSTM.csv", index=False)