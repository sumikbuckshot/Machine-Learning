#!/usr/bin/env python
# coding: utf-8

# In[88]:


import pandas as pd


# In[89]:


#reading csv file from url
df=pd.read_csv('http://13.234.66.67/summer19/datasets/info.csv')


# In[90]:


df.info()


# In[91]:


df


# In[92]:


#removing missing values or repalcing missing values with some relevant data 
df.describe()


# In[93]:


x=df.iloc[:,0:].values
from sklearn.preprocessing import Imputer
x


# In[94]:


imp=Imputer(missing_values='NaN',axis=0,strategy='mean')


# In[95]:


impute=imp.fit(x[:,1:3])


# In[96]:


x[:,1:3]=impute.transform(x[:,1:3])


# In[97]:


X


# In[98]:


#string label int/float
from sklearn.preprocessing import LabelEncoder


# In[99]:


cont=LabelEncoder() 


# In[100]:


#noe apply column first in this label
x[:,0]=cont.fit_transform(x[:,0])


# In[101]:


x


# In[102]:


#labelling last column 
cont1=LabelEncoder()


# In[103]:


x[:,3]=cont1.fit_transform(x[:,3])


# In[104]:


x


# In[105]:


#noe encoding first column  ----- making sub column of first column 
from sklearn.preprocessing import OneHotEncoder


# In[106]:


firstcl=OneHotEncoder(categorical_features=[0])  #defining exact column where we want to make category


# In[107]:


x=firstcl.fit_transform(x).toarray()


# In[110]:


x.astype(int)


# In[ ]:




