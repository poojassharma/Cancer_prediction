#!/usr/bin/env python
# coding: utf-8

# # cancer prediction using KNN 

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt #matplotlib is used for plot the graphs,
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[3]:


#Importing data and display first 5 rows
data=pd.read_csv("C:\\Users\\hp pc\\Desktop\\python\\Datasets\\cancer.csv")
data.head()


# In[4]:


#no of rows and columns of data
data.shape


# In[5]:


data.info()


# In[7]:


sns.lmplot(x ='mean_radius', y ='mean_texture', 
          fit_reg = False, hue = 'diagnosis',
          data = data)
plt.show()


# In[5]:


#checking if there is any null value present or not
data.isnull().sum()


# In[7]:


#finding the pairwise correlation of all columns in the dataframe
corr=data.corr()
corr.nlargest(30,'diagnosis')['diagnosis']


# In[8]:


#creating independent and dependent variable
x=data[['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness']]
y=data[['diagnosis']]


# In[9]:


#Distributing training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# In[10]:


#building KNN model and fitting data into it.
from sklearn.metrics import accuracy_score
model = KNeighborsClassifier(n_neighbors=8)
model.fit(x_train, y_train)
predict = model.predict(x_test)
accuracy_score(predict,y_test)


# In[17]:


#check accuracy of model
print(accuracy_score)


# In[13]:


model.score(x_train,y_train)


# In[14]:


#checking score using cross validation method
from sklearn.model_selection import cross_val_score
score=cross_val_score(model,x,y,cv=2)
score


# In[15]:


model.fit(x_train,y_train)
predict=model.predict(x_test)
predict


# In[16]:


acc = model.score(x_test,y_test)
acc

