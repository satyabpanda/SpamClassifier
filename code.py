#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:36:16 2019

@author: urban
"""
import pandas as pd

messages = pd.read_csv('smsspamcollection/SMSSpamCollection',sep = '\t',names = ["label","messages"])

#data cleaning

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

corpus = []

for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['messages'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

#document matrix
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
X = cv.fit_transform(corpus).toarray()

# convert 2 categorical values

y = pd.get_dummies(messages['label'])
y=y.iloc[:,1].values

#Train test split

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.10,random_state = 0)

#traing Model

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)

y_pred = spam_detect_model.predict(X_test)

#compare y-pred vs y_test

from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test,y_pred)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)




