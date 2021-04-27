#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Check required packages
import numpy as np
import pandas as pd
import seaborn as sns

#sklearn imports
from sklearn.datasets import load_diabetes
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


# In[2]:


#Import Dataset 
diab = load_diabetes()
X = diab['data']
y = diab['target']


# In[3]:


#Data Overview Section
diab_df = pd.DataFrame(diab['data'], columns=diab['feature_names'])
diab_df.head()


# In[4]:


#Data Descriptive Statistic Section
diab_df.describe()


# In[5]:


#Now that we have explored the dataset, we will One Hot encode the sex column 
encoder = preprocessing.OneHotEncoder(categories="auto")
sexMF = encoder.fit_transform(X[:, 1].reshape(-1, 1)).toarray()
print(sexMF[:5, ])
#
#Now we must remove the sex column, and then we will replace it with two different variables 
X = np.delete(X, 1, 1)
X = np.concatenate((sexMF, X), axis=1)
X_df = pd.DataFrame(X)
X_df.head()


# In[6]:


#We can get rid of having both male and female 
X = np.delete(X,1,1)
X


# In[7]:


#Now we can partition our data into an 80/20 split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# In[8]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[9]:


#Creating a model using degree parameters 1-5 on an SVR model. The degree parameter will be aplied to the polynomial kernel 
param_grid = [{'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'degree': [1,2,3,4,5]}]
model = SVR()
svr = GridSearchCV(model,param_grid, cv=5, scoring = 'neg_mean_squared_error') 
#
svr.fit(X_train, y_train)
print(svr.cv_results_)
#
y_pred = svr.predict(X_test)


# In[11]:


#Final Model printouts: Best parameters, MSE, and MAE 
print(svr.best_params_)
print('MSE: {}\nMAE: {}'.format(metrics.mean_squared_error(y_test,y_pred), metrics.mean_absolute_error(y_test,y_pred)))


# In[ ]:




