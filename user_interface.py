import pickle

import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src import config
from src import preprocessing as pp

def predict(text:str):
    """
    Predict the class of an instance
    :param text: The tweet text we want to classify
    :return: The Model Output
    """
    outcome_dict = {0: "Non-Disaster", 1: "Disaster"}

    # path to model
    model_path = f"models/PRETRAIN_WORD2VEC_LSTM/"

    # do cleaning to text
    clean_text = pp.clean_tweet(text)
    clean_text = np.array([clean_text])

    # loading tokenizer
    with open(f'{model_path}tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # convert tokens to sequences and pad them
    data_values = tokenizer.texts_to_sequences(clean_text)
    X_padded = pad_sequences(data_values, maxlen=config.MAXLEN)

    # load the classifier
    clf = load_model(f"{model_path}LSTM_Word2Vec.h5")
    prediction = clf.predict_classes(X_padded, verbose=-1)

    prediction = prediction.sum()
    return outcome_dict[prediction]

if __name__ == "__main__":
    iface = gr.Interface(
        fn=predict,
        inputs= gr.inputs.Textbox(lines=3, placeholder="Insert Tweet..."),
        outputs="text"
    )
    iface.launch()