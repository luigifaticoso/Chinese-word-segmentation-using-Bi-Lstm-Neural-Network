from argparse import ArgumentParser
import importlib
import os
import shutil
from tqdm import tqdm
import keras
import tensorflow as tf
from keras.models import load_model
import numpy as np
import time
from keras import backend as K

classes = {
    0:'B',
    1:"I",
    2:'E',
    3:'S',
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path",help="The path of the input file")
    parser.add_argument("output_path",help="The path of the output file")
    parser.add_argument("resources_path",help="The path of the resources needed to load your model")

    return parser.parse_args()

def load_vocabulary(char_vocabulary_path,bi_vocabulary_path):
    ##
    ##  It loads the vocabularies from the file
    ##

    fv = open(char_vocabulary_path,'r')
    vocabulary_char = {}
    for row in fv.readlines():
        key = row.split(':')[0]
        value = row.split(':')[1].strip()
        vocabulary_char[key]=int(value)
    fv.close()
    fb = open(bi_vocabulary_path,'r')
    vocabulary_bi = {}
    for row in fb.readlines():
        key = row.split(':')[0]
        value = row.split(':')[1].strip()
        vocabulary_bi[key]=int(value)
    fb.close()
    return vocabulary_char,vocabulary_bi
    



def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    
    print("###LOADING MODEL")
    model = tf.keras.models.load_model(os.path.join(resources_path,'model_char_bi.h5'))
    print("###MODEL LOADED")


    print("###LOADING VOCABULARY")
    vocabulary_char,vocabulary_bigram = load_vocabulary(os.path.join(resources_path,'char_vocabulary.txt'),os.path.join(resources_path,'bi_vocabulary.txt'))
    print("###VOCABULARY LOADED")

    ##
    ##  We load the vocabulary and check if all the words in input_path are in it, if they are not they will be replaced with an 'U'
    ##

    fi = open(input_path,'r')
    fo = open(output_path,'w')
    rows = fi.readlines()
    MAX_LENGTH=164
    lista_input1 = list()
    lista_input2 = list()
    lengths_list = list()
    sentences_longer = list()
    for i in tqdm(range(len(rows))):
        input_neural_char=[]
        input_neural_bigram=[]
        list_sentences = rows[i].strip()
        for idx in range(len(list_sentences)):
            if list_sentences[idx] not in vocabulary_char.keys():
               input_neural_char.append(vocabulary_char['U']) 
            else:
                input_neural_char.append(vocabulary_char[list_sentences[idx]])
            if idx < len(list_sentences)-1:
                bigram = list_sentences[idx] + list_sentences[idx+1]
                if bigram not in vocabulary_bigram:
                    input_neural_bigram.append(vocabulary_bigram['U'])
                else:
                    input_neural_bigram.append(vocabulary_bigram[bigram])
            else:
                bigram = list_sentences[idx] + 'E'
                if bigram not in vocabulary_bigram:
                    input_neural_bigram.append(vocabulary_bigram['U'])
                else:
                    input_neural_bigram.append(vocabulary_bigram[bigram])

        length_sentence = len(input_neural_char)

        ##
        ##  This is a workaround to handle the sentences longer than the max_length.
        ##  It splits the sentences in subarrays of length equals to the maxlength and stores the indices.
        ##  In the predicition if an array seems to be a subarray the algorithm will not write a newline
        ##

        extra_array_char = []
        extra_array_bigram = []
        if(length_sentence>MAX_LENGTH):
            temp_index = 1
            while length_sentence > temp_index+MAX_LENGTH-1:
                sentences_longer.append(1)
                start_index = temp_index-1
                end_index = temp_index-1+MAX_LENGTH
                extra_array_char.append(input_neural_char[start_index:end_index])
                extra_array_bigram.append(input_neural_bigram[start_index:end_index])
                lengths_list.append(len(input_neural_char[start_index:end_index]))
                temp_index = end_index

            last_array_char = input_neural_char[temp_index-1:]
            last_array_bi = input_neural_bigram[temp_index-1:]
            lengths_list.append(len(last_array_char)-1)
            while len(last_array_char)<MAX_LENGTH:
                last_array_char.append(0)
                last_array_bi.append(0)
            
            sentences_longer.append(0)
            extra_array_char.append(last_array_char)
            extra_array_bigram.append(last_array_bi)

        else:
            lengths_list.append(len(input_neural_char))
            sentences_longer.append(0)
            while len(input_neural_char)<MAX_LENGTH:
                input_neural_char.append(0)
                input_neural_bigram.append(0)
        
        ##
        ##  here we load the embeddings of the sentence, and in case is longer than max length we will load the subarrays too
        ##
        
        if len(extra_array_char)!=0:
            for k in extra_array_char:
                lista_input1.append(np.array(k))
        else:
            lista_input1.append(np.array(input_neural_char))
        
        if len(extra_array_bigram)!=0:
            for j in extra_array_bigram:
                lista_input2.append(np.array(j))
        else:
            lista_input2.append(np.array(input_neural_bigram))
            
    result = model.predict([np.array(lista_input1),np.array(lista_input2)])

    for o in tqdm(range(len(result))): 

        ##
        ##  0 : B
        ##  1 : I
        ##  2 : E
        ##  3 : S
        ##

        real_len = lengths_list[o]

        for e in range(real_len):
            output = np.argmax(result[o][e])
            fo.write(classes[output])

        ##
        ##  If the array is a subarray it will be denoted in sentence_longer with a value = 1
        ##  If so the predict won't write a new line so to concatenate the prediction
        ##

        if sentences_longer[o]==0:
            fo.write('\n')
    print("that's all folks")
        

if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
    K.clear_session()