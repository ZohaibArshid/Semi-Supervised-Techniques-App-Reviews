# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 23:09:16 2021

@author: Zobi Tanoli
"""

from numpy import argmax
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import nltk
from nltk import word_tokenize
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import heapq
from scipy.stats import entropy


df= pd.read_csv('itor28(highentropy).csv', encoding='Latin1')
#print(df)

Etype=[]

txt = df['Text']
senti =df['Sentiment']
filter_words = []

lemmat = WordNetLemmatizer()
#words to stem
#Stemming the words
for word in txt:
    #lemm = WordNetLemmatizer(stem)
    word_list = nltk.word_tokenize(word)
    lemm =[(lemmat.lemmatize(w)) for w in word_list]
    sent = [' '.join(lemm)] # join list words as a string
    filter_words.append(sent)
        #lemm= lemmat.lemmatize(w)
        #filter_words.append(lemm)
#print(filter_words)

X_train = [item for sublist in filter_words for item in sublist] # make list of lists to single list
#print(len(X_train))
y_train = pd.Series(senti) # convert list to Pandas series
#print(len(y_train))
#print(type(text))
#print(len(text))

#Etype.append(flat_ls)
#Etype.append(senti)

#print(str(Etype))


csvfile = pd.read_csv('sample27.csv', encoding= 'Latin1')
#print(len(csvfile))
smple = csvfile['Text']
#smple= df.sample(200)
#print(smple)
testset= smple.values.tolist()
#print(testset)
#print(len(testset))

# -----------------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words= 'english')
#print(X_train)
X_train = vectorizer.fit_transform(X_train)
#print(X_train)
X_test = vectorizer.transform(smple)
#print(X_test)

feature_names = vectorizer.get_feature_names()
#print(feature_names)

#print(type(X_train),type(X_test),type(y_train),type(y_test))
#print((X_train),(X_test),(y_train),(y_test))
#SVC----> support vector classifier
classifier= svm.SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)
#y_predict = classifier.predict(X_test)
#print(len(y_predict))
#print(X_test)
#print(classifier.score(X_test,y_test))
clp= classifier.predict_proba(X_test)
#pd.DataFrame(classifier.predict_proba(X_test), columns=classifier.classes_)
#print(clp)

################ --------- Technique ---------- ################

problist= clp.tolist()
#print(problist)
maxvalue = []
minusvalue = []

indexing = []
for i in range(300):
    indexing.append(i)
print(indexing)

sent_tuple = list(zip(indexing, testset))
print(sent_tuple)

maxvalue = []
index = []
for i in problist:
    max_value = max(i)
    maxvalue.append(max_value)
    maxindex= i.index(max_value)
    index.append(maxindex)
print(index)
print(maxvalue)
pro_tuple = list(zip(indexing, index, maxvalue))
#print(pro_tuple)

threshld = [x for x in pro_tuple if x[2] > 0.75]
#print(threshld)

my_list = [(a,b,f) for (a,b,c) in threshld for (e,f) in sent_tuple  if (a == e)]
#print(my_list)

index_senti = [x[1] for x in my_list]
print(index_senti)

sentence = [x[2] for x in my_list]
print(sentence)

senti=[]

for i in index_senti:
    if i == 0:
        senti.append('negative')
    elif i == 1:
        senti.append('neutral')
    elif i == 2:
        senti.append('positive')
        
print(len(senti))
data = pd.DataFrame(senti, sentence)
data.to_csv("Selflearning.csv")


