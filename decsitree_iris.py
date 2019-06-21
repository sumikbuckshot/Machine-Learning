#!/usr/bin/env python
# coding: utf-8

# In[27]:


from sklearn.datasets import load_iris
import time
import matplotlib.pyplot as plt


# In[2]:


#loading iris data only
iris=load_iris()


# In[3]:


dir(iris)  #exploring 


# In[5]:


iris.DESCR


# In[7]:


#iris.DESCR     these are features name 
iris.feature_names


# In[8]:


iris.target_names


# In[17]:


#actual data with attributes is 
features=iris.data
#print(features)
print(features.shape)
time.sleep(2)
type(features)


# In[19]:


#now time for target or label data that will be exactly same as number of features data
label=iris.target
label.shape


# In[25]:


SL=features[0:,0]
SW=features[0:,1]


# In[43]:


plt.xlabel("sepal_length")
plt.ylabel("sepal width")
plt.scatter(SL,SW,label="sepal data",marker='.')
plt.scatter(features[0:,2],features[0:,3], label="petal data",marker='.')
plt.legend()


# In[31]:





# In[ ]:




