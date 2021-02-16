from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras import Sequential

from utils import f1_metric

def my_LSTM(embedding_layer):
    print('Creating model...')
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, dropout=0.2,  recurrent_dropout=0.2))
    model.add(Dense(1, activation = "sigmoid"))

    print('Compiling...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=["accuracy"])
    return model