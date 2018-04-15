
# coding: utf-8

# In[5]:

import os
import json
import string
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDClassifier
import sklearn.preprocessing as preprocessing
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
import gensim


# In[6]:

#Build word vector for training set by using the average value of all word vectors in the tweet, then scale
def buildWordVector(text, size, model):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# In[8]:

root_dir = ""

wordnet_lemmatizer = WordNetLemmatizer()


with open(root_dir + 'yelpHeld.json') as data_file:
    yelp = json.load(data_file)


# In[3]:

lda_model = pickle.load(open("lda_model.pickle.dat", "rb"))


# In[10]:

yelp_df = pd.DataFrame(yelp)


# In[12]:

yelp_df_x = yelp_df.drop(columns=['date', 'review_id'])
yelp_df_x['text'] = yelp_df_x['text'].apply(gensim.utils.simple_preprocess)


# In[13]:

X = yelp_df_x.select_dtypes(include=[object]).drop(columns=['text'])
# TODO: create a LabelEncoder object and fit it to each feature in X


# 1. INSTANTIATE
# encode labels with value between 0 and n_classes-1.
le = preprocessing.LabelEncoder()


# 2/3. FIT AND TRANSFORM
# use df.apply() to apply le.fit_transform to all columns
X_2 = X.apply(le.fit_transform)
X_2.columns = ['bid', 'uid']


# In[14]:

yelp_df_x = pd.concat([yelp_df_x, X_2], axis=1)
yelp_df_x = yelp_df_x.drop(columns=['business_id', 'user_id'])


# In[15]:

yelp_df_x.head()


# In[ ]:

#model = pickle.load(open("w2v_model_1.pickle.dat", "rb"))


# In[ ]:

# build vocabulary and train model
model = gensim.models.Word2Vec(
        yelp_df_x['text'],
        size=300,
        window=10,
        min_count=2,
        workers=15)


# In[ ]:

model.train(yelp_df_x['text'], total_examples=len(yelp_df_x), epochs=10)


# In[ ]:

#Build test vectors then scale
test_vecs = np.concatenate([buildWordVector(z, size, model) for z in yelp_df_x['text']])
test_vecs_scaled = scale(test_vecs)


# In[ ]:

yelp_df_x = yelp_df_x.reset_index().drop(columns=['index', 'text'])
yelp_df_x = pd.concat([yelp_df_x, pd.DataFrame(test_vecs_scaled)], axis=1)
pickle.dump(yelp_df_x, open("yelp.pickle.dat", "wb"))


# In[ ]:

predictions = lda_predict(yelp_df_x)


# In[ ]:

pd.DataFrame([yelp_df['review_id'], predictions]).to_csv('stars.csv')
