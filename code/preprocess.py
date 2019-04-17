import os

train_path = 'icwb2-data/training/as_training.utf8'

f = open(train_path,'r')
stripped_sentences = list(map(str.strip, f.readlines()))
# print(stripped_sentences)

bies = {}
for e in stripped_sentences:
    splitted_sentence = e.split("\u3000")
    bies_String = ""
    newstr = e.replace("\u3000","")
    for i in range(len(splitted_sentence)):
        if len(splitted_sentence[i])==1:
            bies_String = bies_String+"S"
        else:
            bies_String = bies_String+"B"
            index = 1
            while(index<len(splitted_sentence[i])-1):
                bies_String = bies_String+"I"
                index+=1
            bies_String = bies_String+"E"
    bies[newstr]= bies_String
print(bies)


