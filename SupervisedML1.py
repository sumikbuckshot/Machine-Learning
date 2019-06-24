#!/usr/bin/env python
# coding: utf-8

# In[15]:


#from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sb


# In[16]:


df=pd.read_csv('http://13.234.66.67/summer19/datasets/diabetest.csv')


# In[17]:


df.info()


# In[18]:


df.describe()


# In[19]:


#printing top five columns 
df.head(5)


# In[20]:


df.tail(4)


# In[ ]:





# In[42]:


# i want to plot a particular column with count
sb.countplot(df['Pregnancies'])


# In[47]:


#df.hist(figsize=(20,15)) #histogram 
#sb.scatterplot(df['Pregnancies'])
sb.scatterplot(df['Pregnancies'],df['Glucose'])


# In[49]:


#extract attribute from dataframe
features=df.iloc[:,0:8].values


# In[50]:


features


# In[53]:


label=df.iloc[:,8].values
label


# In[52]:


label.shape


# In[54]:


#separate training and testing data
from sklearn.model_selection import train_test_split


# In[56]:


X,x,Y,y=train_test_split(features,label,test_size=0.2)
# X is training data
# x is testing data 
# Y is training answer /label
# y is testing answer /label


# In[59]:


#calling decisiontree clf
from sklearn.tree import DecisionTreeClassifier


# In[62]:


#calling
clf=DecisionTreeClassifier()


# In[63]:


trained=clf.fit(X,Y)


# In[65]:


y_predict=trained.predict(x)


# In[66]:


from sklearn.metrics import accuracy_score


# In[68]:


accuracy_score(y,y_predict)


# In[70]:


from sklearn.neighbors import KNeighborsClassifier


# In[71]:


kclf=KNeighborsClassifier()


# In[72]:


kt=kclf.fit(X,Y)


# In[73]:


out1=kt.predict(x)


# In[74]:


accuracy_score(y,out1)


# In[ ]:




