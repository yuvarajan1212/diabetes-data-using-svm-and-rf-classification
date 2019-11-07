#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv(r'C:/Users/CDSS/Desktop/diabetes.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df['Outcome'].value_counts()


# In[7]:


sns.countplot(df['Outcome'])


# In[8]:


#now separate the datset as response variable and feature varible
X = df.drop('Outcome', axis=1)
y = df['Outcome']


# In[9]:


#train and test split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[10]:


#applying standard scaling to get optimized result

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[11]:


X_train[:10]


# In[12]:


#Random Forest Classifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)


# In[13]:


pred = rfc.predict(X_test)


# In[14]:


#lets's see how our model perform

print(classification_report(y_test, pred))

print(confusion_matrix(y_test,pred))


# In[15]:


#svm
clf = svm.SVC()
clf.fit(X_train,y_train)
pred1 = clf.predict(X_test)


# In[16]:


print(classification_report(y_test, pred1))
print(confusion_matrix(y_test,pred1))


# In[ ]:




