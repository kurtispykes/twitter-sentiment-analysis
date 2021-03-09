from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras import Sequential

def my_LSTM(embedding_layer):
    print('Creating model...')
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units=64, dropout=0.1,  recurrent_dropout=0.1)))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation = "sigmoid"))

    print('Compiling...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=["accuracy"])
    return model