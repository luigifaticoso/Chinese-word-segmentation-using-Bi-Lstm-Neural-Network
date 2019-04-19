import tensorflow as tf
import tensorflow.keras as K
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_model(vocab_size, hidden_size):


    print("Creating KERAS model")
    model = K.models.Sequential()
    # remember to set mask_zero=True or the model consider the padding as a valid timestep!
    model.add(K.layers.Embedding(len(vocabolario), MAX_LENGTH, mask_zero=True))
    #add a LSTM layer with some dropout in it
    model.add(K.layers.Bidirectional(K.layers.LSTM(20, return_sequences=True,dropout=0.2, recurrent_dropout=0.2,), input_shape=(MAX_LENGTH, 1)))
    model.add(K.layers.Bidirectional(K.layers.LSTM(20, return_sequences=False,dropout=0.2, recurrent_dropout=0.2,), input_shape=(MAX_LENGTH, 1)))
    # add a dense layer with sigmoid to get a probability value from 0.0 to 1.0
    model.add(K.layers.Dense(MAX_LENGTH, activation='softmax'))

    # we are going to use the Adam optimizer which is a really powerful optimizer.
    model.compile(loss='binary_crossentropy', optimizer=K.optimizers.Adam(), metrics=['acc'])

    return model
