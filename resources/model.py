import tensorflow as tf
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Input
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding

from keras.layers import concatenate
from keras.models import Model

def create_model(len_vocabulary_char,len_vocabulary_bigram, hidden_size,MAX_LENGTH,train_char_shape,train_bigram_shape):
    print("char size: {} \n bigram size: {}".format(len_vocabulary_char,len_vocabulary_bigram))
    # define two sets of inputs
    inputChar = Input(shape=(MAX_LENGTH,))
    inputBigram = Input(shape=(MAX_LENGTH,))
    
    # the first branch operates on the first input
    x = Embedding(len_vocabulary_char, MAX_LENGTH, mask_zero=True)(inputChar)
    x = Model(inputs=inputChar, outputs=x)
    
    # the second branch opreates on the second input
    y = Embedding(len_vocabulary_bigram, MAX_LENGTH, mask_zero=True)(inputBigram)
    y = Model(inputs=inputBigram, outputs=y)
    
    # combine the output of the two branches
    combined = concatenate([x.output, y.output])
    
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Bidirectional(LSTM(hidden_size, return_sequences=True,dropout=0.2, recurrent_dropout=0.2,), input_shape=(MAX_LENGTH, 1))(combined)
    z = Dense(4, activation='softmax')(z)
    
    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[x.input, y.input], outputs=z)

    return model
