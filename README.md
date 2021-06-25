# semi-supervised-techniques

Libraries needed:
'''
from numpy import argmax

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import svm


from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report

import nltk

from nltk import word_tokenize

nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

import heapq

from scipy.stats import entropy
'''

we have "sample.py" file we get sample from pool of unlabeled data.

"selftraining.py" file applying selftraining technique and put all seleted reviews in "selflearning.csv". 

"leastcp.py" file applying leastcp strategy and put all selected reviews in "leastcp.csv"

"smallmargin.py" file applying small margin strategy and put all selected reviews in "smallmargin.csv"

"highentropy.py" file applying high entropy strategy and put all selected reviews in "highentropy.csv"

"modelevaluation.py" file used for calculating results.
