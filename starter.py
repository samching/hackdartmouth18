import os
import json
import string
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

root_dir = "."
    
wordnet_lemmatizer = WordNetLemmatizer()


with open(root_dir + 'yelp.json') as data_file:    
    yelp = json.load(data_file)