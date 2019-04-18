import tensorflow as tf
import tensorflow.keras as K
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_model(vocab_size, hidden_size):

    print("Creating KERAS model")
    model = Sequential()

    model.add(Embedding(len(vocabolario), vocab_size, mask_zero=True))
    #add a LSTM layer with some dropout in it
    model.add(Bidirectional(LSTM(hidden_size, return_sequences=False,dropout=0.2, recurrent_dropout=0.2,), input_shape=(vocab_size, 1)))
    # add a dense layer with softmax
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    # Let's print a summary of the model
    model.summary()

    cbk = K.callbacks.TensorBoard("logging/keras_model")
    print("\nStarting training...")

    return model
