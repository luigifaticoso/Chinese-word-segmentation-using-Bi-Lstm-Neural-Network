import tensorflow as tf
import tensorflow.keras as K
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from preprocess import make_dataset
from model import create_model
from keras.optimizers import Adam




train_x_char,train_x_bigram,train_y,test_x_char,test_x_bigram,test_y,dev_x_char,dev_y_char,dev_x_bigram,dev_y_bigram,char_vocabulary,bigram_vocabulary,MAX_LENGTH = make_dataset('output.utf8','dataset.txt')

##
##  I am writing the vocabulary into a file so that i can load it in the prediction file to check it against the predict input file
##

batch_size = 64
epochs = 3
hidden_size = 100
embedding_size_char = 32
embedding_size_bigram = 64
fv = open('char_vocabulary.txt','w')
fb = open('bi_vocabulary.txt','w')
for word in char_vocabulary:
    fv.write(word+":"+str(char_vocabulary[word])+'\n')
for word in bigram_vocabulary:
    fb.write(word+":"+str(bigram_vocabulary[word])+'\n')       
model = create_model(len(char_vocabulary),len(bigram_vocabulary),hidden_size,MAX_LENGTH,train_x_char.shape,train_x_bigram.shape)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
# Let's print a summary of the model
model.summary()

cbk = K.callbacks.TensorBoard("logging/keras_model")
print("\nStarting training...")
    # train the model
print("[INFO] training model...")
model.fit(
	[train_x_char, train_x_bigram], train_y,
	validation_data=([dev_x_char, dev_x_bigram], dev_y_char),
    epochs=epochs, batch_size=batch_size,callbacks=[cbk])
print("Training complete.\n")


model.save("model_char_bi.h5")


print("Saved model to disk")
print("\nEvaluating test...")
loss_acc = model.evaluate(test_x, test_y, verbose=0)
print("Test data: loss = %0.6f  accuracy = %0.2f%% " % (loss_acc[0], loss_acc[1]*100))


print("\nEvaluating test...")
loss_acc = model.evaluate(test_x, test_y, verbose=0)
print("Test data: loss = %0.6f  accuracy = %0.2f%% " % (loss_acc[0], loss_acc[1]*100))



