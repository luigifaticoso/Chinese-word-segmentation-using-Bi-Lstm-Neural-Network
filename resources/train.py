import tensorflow as tf
import tensorflow.keras as K
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Embedding
from preprocess import make_dataset
from model import create_model





train_x,train_y,test_x,test_y,dev_x,dev_y,vocabolary = make_dataset('output.utf8','dataset.txt')
model = create_model(len(vocabolary),20)
batch_size = 32
epochs = 3
# Let's print a summary of the model
model.summary()

cbk = K.callbacks.TensorBoard("logging/keras_model")
print("\nStarting training...")
model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
          shuffle=True, validation_data=(dev_x, dev_y), callbacks=[cbk]) 
print("Training complete.\n")

print("\nEvaluating test...")
loss_acc = model.evaluate(test_x, test_y, verbose=0)
print("Test data: loss = %0.6f  accuracy = %0.2f%% " % (loss_acc[0], loss_acc[1]*100))




