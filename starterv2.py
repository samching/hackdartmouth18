
# coding: utf-8

# In[1]:

import os
import json
import string
import nltk


# In[2]:

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


# In[3]:

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[4]:

from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
import gensim


# In[5]:

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
size = 300


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


# In[7]:

def getTime(date):
    dates = date.split('-')
    total = 0
    total += (int(dates[0])-1970)*3.154e7
    total += (int(dates[1])-1)*2.628e6
    total += (int(dates[2])-1)*86400
    return total


# In[8]:

root_dir = ""

wordnet_lemmatizer = WordNetLemmatizer()


with open(root_dir + 'yelp.json') as data_file:
    yelp = json.load(data_file)


# In[9]:

yelp_df = pd.DataFrame(yelp)


# In[10]:

yelp_df_subset = yelp_df.iloc[1:100000]


# ## Train - Test Split

# In[11]:

yelp_df_y = yelp_df_subset['stars']
yelp_df_x = yelp_df_subset.drop(columns=['stars', 'date', 'review_id'])
#yelp_df_x['date'] = scale(yelp_df_x['date'].apply(getTime))
yelp_df_x['text'] = yelp_df_x['text'].apply(gensim.utils.simple_preprocess)


# In[12]:

# limit to categorical data using df.select_dtypes()
X = yelp_df_x.select_dtypes(include=[object]).drop(columns=['text'])


# In[13]:

# TODO: create a LabelEncoder object and fit it to each feature in X


# 1. INSTANTIATE
# encode labels with value between 0 and n_classes-1.
le = preprocessing.LabelEncoder()


# 2/3. FIT AND TRANSFORM
# use df.apply() to apply le.fit_transform to all columns
X_2 = X.apply(le.fit_transform)
X_2.columns = ['bid', 'rid', 'uid']


# In[14]:

yelp_df_x = pd.concat([yelp_df_x, X_2], axis=1)
yelp_df_x = yelp_df_x.drop(columns=['business_id', 'user_id'])


# In[15]:

yelp_df_x.head()


# In[16]:

# Split into Train / Test set
x_train, x_test, y_train, y_test = train_test_split(yelp_df_x, yelp_df_y, test_size=0.2)


# In[17]:

# build vocabulary and train model
model = gensim.models.Word2Vec(
        x_train['text'],
        size=size,
        window=10,
        min_count=2,
        workers=10)


# In[18]:

model.train(x_train['text'], total_examples=len(x_train), epochs=10)


# In[19]:

train_vecs = np.concatenate([buildWordVector(z, size, model) for z in x_train['text']])
train_vecs_scaled = scale(train_vecs)


# In[20]:

model.train(x_test['text'], total_examples=len(x_test), epochs=10)


# In[21]:

x_train = x_train.reset_index().drop(columns=['index', 'text'])
x_train = pd.concat([x_train, pd.DataFrame(train_vecs_scaled)], axis=1)
y_train = y_train.reset_index().drop(columns=['index']).values.ravel()


# In[22]:

#Build test vectors then scale
test_vecs = np.concatenate([buildWordVector(z, size, model) for z in x_test['text']])
test_vecs_scaled = scale(test_vecs)


# In[23]:

x_test = x_test.reset_index().drop(columns=['index', 'text'])
x_test = pd.concat([x_test, pd.DataFrame(test_vecs_scaled)], axis=1)
y_test = y_test.reset_index().drop(columns=['index']).values.ravel()


# In[33]:

lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(x_train, y_train)


# In[34]:

print ('Test Accuracy: {}'.format(lr.score(x_test, y_test)))


# In[ ]:

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring, n_jobs=-1)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[ ]:

# Compare Algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()
