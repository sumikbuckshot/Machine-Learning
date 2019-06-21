#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np 
import pandas as pd
import matplotlib 
from sklearn.tree import DecisionTreeClassifier


# In[4]:


features=[[100,0],[120,0],[130,1],[150,0]]


# In[5]:


label=["apple","apple","orange","orange"]


# In[25]:


clf=DecisionTreeClassifier()
trained=clf.fit(features,label)
output=trained.predict([[125.1,0]])
print(output)


# In[10]:


get_ipython().system('pip3 install scikit-learn')


# In[ ]:




