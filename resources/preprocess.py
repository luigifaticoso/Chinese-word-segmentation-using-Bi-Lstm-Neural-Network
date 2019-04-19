from tqdm import tqdm
import os

from tqdm import tqdm
vocabolario = {}
dataset = "output.utf8"
f = open(dataset,'r')
fw = open("training.txt","w+")
listafrasi = f.readlines()
count = 0
for frase in tqdm(range(len(listafrasi))):
  frase_new = listafrasi[frase].strip()
  for e in frase_new:
    if e not in vocabolario.keys():
      vocabolario[e] = count
      count+=1
  parola_dict = ""
  frase_splitted = frase_new.split("\u3000")
  frase_dict = frase_new.replace("\u3000","")
  for parola in frase_splitted:
    if len(parola) == 1:
      parola_dict+='s'
    elif len(parola) > 1:
      parola_dict+='b'
      for lunghezza in range(len(parola)-2):
        parola_dict+='i'
      parola_dict+='e'
      
  fw.write(frase_dict + '\t' + parola_dict + '\n')
  #dizionario_frasi[frase_dict] = parola_dict
# print(vocabolario)
        
        
import tensorflow as tf
import tensorflow.keras as K
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
#DEFINE SOME COSTANTS
MAX_LENGTH = 0
EMBEDDING_SIZE = 32
HIDDEN_SIZE = 100

#(train_x,train_y)

fl = open('training.txt','r')

frasi = fl.readlines()
lista_frasi_train = list()
lista_frasi_test = list()
lista_label_train = list()
lista_label_test = list()
# train_x = np.zeros((len(frasi), MAX_LENGTH))
count = 1
for idx in tqdm(range(len(frasi))):
  frase_splitted = frasi[idx].split('\t')
  frase_words = frase_splitted[0]
  label = frase_splitted[1].strip()
  frase_nums = list()
  label_nums = list()
  MAX_LENGTH = max(MAX_LENGTH,len(frase_words))
  count+=1
  for i in frase_words:
    frase_nums.append(vocabolario[i])

  for l in label:
      label_nums.append(l)

  if count < (70*len(frasi))/100:
    lista_frasi_train.append(frase_nums)
    lista_label_train.append(label_nums)
  else:
    lista_frasi_test.append(frase_nums)
    lista_label_test.append(label_nums)


train_x = np.asarray(lista_frasi_train)
train_y = np.asarray(lista_label_train)
test_x = np.asarray(lista_frasi_test)
test_y = np.asarray(lista_label_test)

print("##### START PADDING")
# When truncating, get rid of initial words and keep last part of the review. (longer sentences)
# When padding, pad at the end of the sentence. (shorter sentences)
train_x = pad_sequences(train_x, truncating='pre', padding='post', maxlen=MAX_LENGTH)
train_y = pad_sequences(train_x, truncating='pre', padding='post', maxlen=MAX_LENGTH)

test_x = pad_sequences(test_x, truncating='pre', padding='post', maxlen=MAX_LENGTH)
test_y = pad_sequences(train_x, truncating='pre', padding='post', maxlen=MAX_LENGTH)
print("train_x shape {}, train_y shape {}".format(train_x.shape,train_y.shape))
print("##### END PADDING")


print("##### START SPLITTING")      
# Take 5% of the training set and use it as dev set
# stratify makes sure that the development set follows the same distributions as the training set:
# half positive and half nevative.
train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=.05, random_state=42)
print("##### END SPLITTING") 

print("Training set shape:", train_x.shape)
print("Dev set shape:", dev_x.shape)
print("Test set shape:", test_x.shape)

print("Creating KERAS model")
model = Sequential()
# remember to set mask_zero=True or the model consider the padding as a valid timestep!
model.add(Embedding(len(vocabolario), 188, mask_zero=True))
#add a LSTM layer with some dropout in it
model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=False,dropout=0.2, recurrent_dropout=0.2,), input_shape=(188, 1)))
# add a dense layer with sigmoid to get a probability value from 0.0 to 1.0
model.add(Dense(188, activation='softmax'))

# we are going to use the Adam optimizer which is a really powerful optimizer.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
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


def make_dataset(input_path,output_path):
  vocabolario = {}
  dataset = input_path
  f = open(dataset,'r')
  fw = open(output_path,"w+")
  frasi = fl.readlines()
  lista_frasi_train = list()
  lista_frasi_test = list()
  lista_label_train = list()
  lista_label_test = list()
  lunghezza_frasi = list()
  # train_x = np.zeros((len(frasi), MAX_LENGTH))
  count = 1
  for idx in tqdm(range(len(frasi))):
    frase_splitted = frasi[idx].split('\t')
    frase_words = frase_splitted[0]
    label = frase_splitted[1].strip()
    frase_nums = list()
    label_nums = list()
    #TODO:  fare la media
  #   lunghezza_frasi.append(len(frase_words))
    MAX_LENGTH = max(MAX_LENGTH,len(frase_words))
    count+=1
    for i in frase_words:
      frase_nums.append(vocabolario[i])

    for l in label:
      if l == 'b':
        label_nums.append(1)
      elif l == 'i':
        label_nums.append(2)
      elif l == 'e':
        label_nums.append(3)
      elif l == 's':
        label_nums.append(4)
          
        

    if count < (70*len(frasi))/100:
      lista_frasi_train.append(frase_nums)
      lista_label_train.append(label_nums)
      print(frase_nums,label_nums)
      exit()
    else:
      lista_frasi_test.append(frase_nums)
      lista_label_test.append(label_nums)

  # MAX_LENGTH = int(sum(lunghezza_frasi)/len(lunghezza_frasi))
  # MAX_LENGTH = int(MAX_LENGTH/2)
  train_x = np.asarray(lista_frasi_train)
  train_y = np.asarray(lista_label_train)
  test_x = np.asarray(lista_frasi_test)
  test_y = np.asarray(lista_label_test)

  print("##### START PADDING")
  # When truncating, get rid of initial words and keep last part of the review. (longer sentences)
  # When padding, pad at the end of the sentence. (shorter sentences)
  train_x = pad_sequences(train_x, truncating='pre', padding='post', maxlen=MAX_LENGTH)
  train_y = pad_sequences(train_y, truncating='pre', padding='post', maxlen=MAX_LENGTH)

  test_x = pad_sequences(test_x, truncating='pre', padding='post', maxlen=MAX_LENGTH)
  test_y = pad_sequences(train_x, truncating='pre', padding='post', maxlen=MAX_LENGTH)
  print("train_x shape {}, train_y shape {}".format(train_x.shape,train_y.shape))
  print("##### END PADDING")


  print("##### START SPLITTING")      
  # Take 5% of the training set and use it as dev set
  # stratify makes sure that the development set follows the same distributions as the training set:
  # half positive and half nevative.
  train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=.05)
  print("##### END SPLITTING") 

  print("Training set shape:", train_x.shape)
  print("Dev set shape:", dev_x.shape)
  print("Test set shape:", test_x.shape)


  return train_x,train_y,test_x,test_y,dev_x,dev_y,vocabolary
