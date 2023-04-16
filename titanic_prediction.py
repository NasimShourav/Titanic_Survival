#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries

#data analysis libraries 
import numpy as np
import pandas as pd

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#importing dataframe

titanic_df = pd.read_csv("C:\\D Drive\\CSE\\Data_Science\\Project\\practice\\titanic_survival\\tested.csv")


# In[3]:


titanic_df.head()


# In[4]:


titanic_df = titanic_df.drop(['Cabin'], axis=1)


# In[5]:


titanic_df.head()


# In[6]:


titanic_df[['Last_Name', 'First_Name']] = titanic_df.Name.str.split(",", expand = True)


# In[7]:


titanic_df.head()


# In[8]:


titanic_df['First_Name'].str.strip()


# In[9]:


titanic_df.head()


# In[10]:


titanic_df[['Title', 'First Name']] = titanic_df.First_Name.str.split(".", expand = True)


# In[11]:


titanic_df.head()


# In[12]:


titanic_df = titanic_df.drop(['First Name', 'Name', 'Last_Name', 'First_Name'], axis=1)


# In[13]:


titanic_df.head()


# In[20]:


titanic_df.isnull().sum()


# In[58]:


titanic_df2=titanic_df.dropna(subset=['Fare'])


# In[59]:


titanic_df2.isnull().sum()


# In[60]:


titanic_df2.groupby('Title').size().sort_values(ascending=False)


# In[61]:


result = titanic_df2.groupby('Title').agg({'Age': ['mean', 'min', 'max']})
result


# In[62]:


import random
titanic_df2.loc[((titanic_df2.Title ==' Col')
         & (titanic_df2['Age'].isnull())), 'Age'] = random.randint(47,53)
titanic_df2.loc[((titanic_df2.Title ==' Dona')
         & (titanic_df2['Age'].isnull())), 'Age'] = random.randint(39,39)
titanic_df2.loc[((titanic_df2.Title ==' Dr')
         & (titanic_df2['Age'].isnull())), 'Age'] = random.randint(53,53)
titanic_df2.loc[((titanic_df2.Title ==' Master')
         & (titanic_df2['Age'].isnull())), 'Age'] = random.randint(1,14)
titanic_df2.loc[((titanic_df2.Title ==' Miss')
         & (titanic_df2['Age'].isnull())), 'Age'] = random.randint(0,45)
titanic_df2.loc[((titanic_df2.Title ==' Mr')
         & (titanic_df2['Age'].isnull())), 'Age'] = random.randint(14,67)
titanic_df2.loc[((titanic_df2.Title ==' Mrs')
         & (titanic_df2['Age'].isnull())), 'Age'] = random.randint(16,76)
titanic_df2.loc[((titanic_df2.Title ==' Rev')
         & (titanic_df2['Age'].isnull())), 'Age'] = random.randint(30,41)


# In[63]:


titanic_df2


# In[64]:


titanic_df2.isnull().sum()


# In[69]:


titanic_df3 = titanic_df2.dropna()


# In[70]:


titanic_df3


# In[72]:


titanic_df3.groupby('Title').size().sort_values(ascending=False)


# In[73]:


titanic_df3


# In[79]:


titanic_df4 = titanic_df3.drop(['PassengerId', 'Ticket','Title'], axis=1)


# In[102]:


import seaborn as sns

sns.barplot(x="Parch", y="Survived", data=titanic_df4)


# In[103]:


sns.barplot(x="Sex", y="Survived", data=titanic_df4)


# In[104]:


sns.barplot(x="Pclass", y="Survived", data=titanic_df4)


# In[105]:


sns.barplot(x="Embarked", y="Survived", data=titanic_df4)


# In[80]:


titanic_df4


# In[110]:


titanic_df4.loc[titanic_df100['Age']<=19, 'age_group'] = 'teenage'
titanic_df4.loc[titanic_df100['Age'].between(20,24), 'age_group'] = 'yadult'
titanic_df4.loc[titanic_df100['Age'].between(25,39), 'age_group'] = 'adult'
titanic_df4.loc[titanic_df100['Age']>39, 'age_group'] = 'older_adult'


# In[112]:


sns.barplot(x="age_group", y="Survived", data=titanic_df4)


# In[94]:


titanic_df5 = pd.get_dummies(titanic_df4.Embarked, prefix='Embarked')


# In[95]:


titanic_df5


# In[96]:


titanic_df6 = pd.get_dummies(titanic_df4.Sex, prefix='Sex')


# In[97]:


titanic_df6


# In[98]:


titanic_df_final = pd.concat([titanic_df4, titanic_df5, titanic_df6], axis=1, join="inner")


# In[99]:


titanic_df_final


# In[100]:


titanic_df_final = titanic_df_final.drop(['Sex', 'Embarked'], axis=1)


# In[101]:


titanic_df_final


# In[113]:


X = titanic_df_final.drop(['Survived'], axis=1)
y = titanic_df_final.Survived


# In[114]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


# In[116]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)
acc_gaussian = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_gaussian)


# In[119]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_logreg)


# In[120]:


# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc_svc = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_svc)


# In[121]:


# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_linear_svc)


# In[122]:


# Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_test)
acc_perceptron = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_perceptron)


# In[123]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(X_train, y_train)
y_pred = decisiontree.predict(X_test)
acc_decisiontree = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_decisiontree)


# In[124]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(X_train, y_train)
y_pred = randomforest.predict(X_test)
acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_randomforest)


# In[125]:


# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_knn = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_knn)


# In[126]:


# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)
acc_sgd = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_sgd)


# In[127]:


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(X_train, y_train)
y_pred = gbk.predict(X_test)
acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_gbk)


# In[130]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)


# In[ ]:




