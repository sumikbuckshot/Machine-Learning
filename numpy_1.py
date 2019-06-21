
# coding: utf-8

# In[2]:


import numpy as np


# In[9]:


print("enter no of rows: ")
a=int(input())
print ("enter no of columns : ")
b=int(input())
print("your array is of dimensions :",a,"x",b)
values=np.random.rand(a,b)
print(values)

np.savetxt('numpy1.txt',values)
np.loadtxt('numpy1.txt')


# ##### 

# In[7]:




