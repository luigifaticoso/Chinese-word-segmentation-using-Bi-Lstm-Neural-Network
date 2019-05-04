from tqdm import tqdm
import os
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import time

classes = {
  'B' : np.array([1,0,0,0]),
  'I' : np.array([0,1,0,0]),
  'E' : np.array([0,0,1,0]),
  'S' : np.array([0,0,0,1]),
}

def make_dataset(input_path,output_path):

  ##
  ##  Here we loop through the sentences to labelling every word
  ##  we add every word we don't know to the dictionary giving it a sequential integer identifier
  ##

  char_vocabulary = {}
  bigram_vocabulary = {}
  ##  This is used to handle unknown char in the test set
  bigram_vocabulary['U'] = 1
  char_vocabulary['U'] = 1
  f = open(input_path,'r')
  fw = open(output_path,"w")
  sentence_list = f.readlines()
  count_char = 2
  count_bigram = 2
  for sentence in tqdm(range(len(sentence_list))):
    sentence_new = sentence_list[sentence].strip()
    sentence_voc = sentence_new.replace("\u3000","")
    for e in range(len(sentence_voc)):
      if sentence_voc[e] not in char_vocabulary.keys():
        count_char+=1
        char_vocabulary[sentence_voc[e]] = count_char
      if e != len(sentence_voc)-1:
        bigram = sentence_voc[e]+sentence_voc[e+1]
        if bigram not in bigram_vocabulary.keys():
          count_bigram +=1
          bigram_vocabulary[bigram] = count_bigram
      else:
        bigram = sentence_voc[e]+"E"
        if bigram not in bigram_vocabulary.keys():
          count_bigram+=1
          bigram_vocabulary[bigram] = count_bigram

    word_dict = ""
    sentence_splitted = sentence_new.split("\u3000")
    for word in sentence_splitted:
      if len(word) == 1:
        word_dict+='S'
      elif len(word) > 1:
        word_dict+='B'
        for lunghezza in range(len(word)-2):
          word_dict+='I'
        word_dict+='E'

    
    fw.write(sentence_voc + '\t' + word_dict + '\n')

  ##
  ##  Preprocessing first part ended.
  ##


  ##
  ##  Here I look to set the MAX_LENGHT so we loop thought the sentences and pick the maximum among all the lenghts
  ##

  MAX_LENGTH = 0
  fl = open(output_path,'r')
  sentences = fl.readlines()
  for i in sentences: 
    sentence_splitted = i.split('\t')
    sentence_words = sentence_splitted[0]
    MAX_LENGTH = max(MAX_LENGTH,len(sentence_words))
  print("Maximum lenght is: {}".format(MAX_LENGTH))
  
  lista_sentences_train_char = list()
  lista_sentences_train_bigram = list()
  lista_sentences_test_char = list()
  lista_sentences_test_bigram = list()
  lista_label_train_char = list()
  lista_label_test_char = list()

  ##
  ##  For the second part of the training i am looking to create the set used for the neural network  
  ##

  for idx in tqdm(range(len(sentences))):
    sentence_splitted = sentences[idx].split('\t')
    sentence_words = sentence_splitted[0]
    label = sentence_splitted[1].strip()
    sentence_nums_char = []
    sentence_nums_bigram = []
    label_nums = []
    for i in range(len(sentence_words)):
      sentence_nums_char.append(char_vocabulary[sentence_words[i]])
      if i < len(sentence_words)-1:
        sentence_nums_bigram.append(bigram_vocabulary[sentence_words[i]+sentence_words[i+1]])
      else:
        sentence_nums_bigram.append(bigram_vocabulary[sentence_words[i]+'E'])

    if(len(sentence_nums_char)!=len(sentence_nums_bigram)):
      print('mismatch')
    for l in range(len(label)):
      label_nums.append(classes[label[l]])
      

    ##
    ##  Following is the padding used for the y set in which i have added [0,0,0,0] until reaching the MAX_LENGTH
    ##

    while(len(label_nums)<MAX_LENGTH):
      label_nums.append(np.array([0,0,0,0]))

    ##
    ##  Here i am diving the set in 80% for the training and rest for the testing
    ##

    if idx < (80*(len(sentences))/100):
      lista_sentences_train_char.append(sentence_nums_char)
      lista_sentences_train_bigram.append(sentence_nums_bigram)
      lista_label_train_char.append(np.array(label_nums))
    else:
      lista_sentences_test_char.append(sentence_nums_char)
      lista_sentences_test_bigram.append(sentence_nums_bigram)
      lista_label_test_char.append(np.array(label_nums))

  train_x_char = lista_sentences_train_char
  train_x_bigram = lista_sentences_train_bigram
  train_y = np.array(lista_label_train_char)

  test_x_char = lista_sentences_test_char
  test_x_bigram = lista_sentences_test_bigram
  test_y = np.array(lista_label_test_char)

  ##
  ##  Here we pad the x files until reaching the MAX_LENGTH
  ##

  print("##### START PADDING")
  train_x_char = pad_sequences(train_x_char, truncating='pre', padding='post', maxlen=MAX_LENGTH)
  train_x_bigram = pad_sequences(train_x_bigram, truncating='pre', padding='post', maxlen=MAX_LENGTH)
  test_x_char = pad_sequences(test_x_char, truncating='pre', padding='post', maxlen=MAX_LENGTH)
  test_x_bigram = pad_sequences(test_x_bigram, truncating='pre', padding='post', maxlen=MAX_LENGTH)
  print("##### END PADDING")

  ##
  ##  Take 5% of the training set and use it as dev set
  ##

  print("train x bigram shape: {}, train_y shape: {}".format(train_x_bigram.shape,train_y.shape))
  print("##### START SPLITTING")      
  train_x_char, dev_x_char, _, dev_y_char = train_test_split(train_x_char, train_y, test_size=.05)

  train_x_bigram, dev_x_bigram, train_y_new, dev_y_bigram = train_test_split(train_x_bigram, train_y, test_size=.05)

  print("##### END SPLITTING") 

  print("Training_x set shape:", train_x_char.shape)
  print("Training_x bigram set shape:", train_x_bigram.shape)


  return train_x_char,train_x_bigram,train_y_new,test_x_char,test_x_bigram,test_y,dev_x_char,dev_y_char,dev_x_bigram,dev_y_bigram,char_vocabulary,bigram_vocabulary,MAX_LENGTH
