# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 23:01:21 2021

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



df= pd.read_csv('itor29(smallmargin).csv', encoding='Latin1')
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

csvfile = pd.read_csv('sample28.csv', encoding= 'Latin1')
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


classifier= svm.SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)

clp= classifier.predict_proba(X_test)

################ --------- Technique ---------- ################

problist= clp.tolist()
#print(problist)
maxvalue = []
minusvalue = []



indexing = []
for i in range(300):
    indexing.append(i)
#print(indexing)

sent_tuple = list(zip(indexing, testset))
#print(sent_tuple)

small_margin = []

for i in problist:
    largest_integers = heapq.nlargest(2, i) 
    largest_integer = largest_integers[0]  # 39
    #print(largest_integer)
    second_largest_integer = largest_integers[1] # 26
    #print(second_largest_integer)
    number = largest_integer - second_largest_integer
    small_margin.append(number)

pro_tuple = list(zip(small_margin, indexing))
pro_tuple.sort()
print(pro_tuple)

my_list = [(a,d) for (a,b) in pro_tuple for (c,d) in sent_tuple  if (b == c)]
print(len(my_list))

k = 40
res = my_list[:k]
print(res)


index_senti = [x[1] for x in res]
print(index_senti)

data = pd.DataFrame(index_senti)
data.to_csv('smallmargin.csv')

