#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 20:25:45 2020

@author: ajeeshsunny
"""
##Stroke prediction

import pandas as pd
url = 'https://raw.githubusercontent.com/BBalajadia/McKinseyAOHack_HealthcareAnalytics/master/data/train.csv'
train = pd.read_csv(url, error_bad_lines=False)

url1 = 'https://raw.githubusercontent.com/BBalajadia/McKinseyAOHack_HealthcareAnalytics/master/data/test.csv'
test = pd.read_csv(url1, error_bad_lines=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.metrics import auc,roc_auc_score,precision_score,recall_score
from sklearn.metrics import classification_report

#checking missing data

train.isnull().sum()

#Data preprocessing drop all missing

train_data = train.dropna(axis=0, how = "any")
test_data = test.dropna(axis=0, how = "any")

train_data['stroke'].value_counts()

sns.countplot(x=train_data["stroke"])
plt.title("No. of patients affected by stroke", fontsize =15)
plot.show()

train_data['gender'].value_counts()

sns.countplot(x=train_data["gender"])
plt.title("No. of patients gender", fontsize =15)
plot.show()

train_data.groupby(['gender'])['stroke'].value_counts()

sns.countplot(x=train_data["gender"], hue = train_data["stroke"])
plt.title("gender vs stroke", fontsize =15)
plt.show()

str_data = train_data.select_dtypes(include=['object'])
str_dt = test_data.select_dtypes(include=['object'])

int_data = train_data.select_dtypes(include=['integer', 'float'])
int_dt = test_data.select_dtypes(include=['integer', 'float'])

##Label Encoder

label = LabelEncoder()
features= str_data.apply(label.fit_transform)
features=features.join(int_data)
features.head()

test1 = str_dt.apply(label.fit_transform)
Test = test1.join(int_dt)
Test.head()

xtrain = features.drop(['stroke'], axis = 1)
ytrain = features["stroke"]
ytrain.head()

#splitting the data into training and testing

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xtrain, ytrain)

##Building NaiveBayes Model

model=GaussianNB()
model.fit(x_train, y_train)

predict=model.predict(x_test)
predict

test_score=model.score(x_test, y_test)
print('NBtest_score:', test_score)

train_score=model.score(x_train, y_train)
print('NBtest_score:', train_score)

##Cross Validation

from sklearn.model_selection import cross_validate
cv_results = cross_validate(model, xtrain, ytrain, cv=5)
cv_results

#NaiveBayes Confusion Matrix

nb_conf_mtr = pd.crosstab(y_test, predict)
nb_conf_mtr

#Classification Report for naivebayes

nbreport = classification_report(y_test, predict)
print(nbreport)

##Building Decision Tree Model

dt_mod =DecisionTreeClassifier(criterion='entropy', max_depth=8)
dt_mod.fit(x_train, y_train)
y_pred=dt_mod.predict(x_test)
y_pred

ts_dt_score = dt_mod.score(x_test, y_test)
print('DTtest_score:', ts_dt_score)

#Classification Report for Decision tree

dt_report = classification_report(y_test, y_pred)
print(dt_report)

import pickle

with open('model_pickle', 'wb') as f:
    pickle.dump(model,f)

pickle_out = open("NB_model.pkl", 'wb')
pickle.dump(model, pickle_out)
pickle_out.close()