#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[2]:


sns.get_dataset_names()


# In[4]:


ds=sns.load_dataset("penguins")
ds


# In[14]:


ds.isnull().sum()


# In[15]:


ds.interpolate(method ="linear",inplace=True)


# In[17]:


ds.isnull().sum()


# In[18]:


ds.species.unique()   #catogorical data 


# In[19]:


ds.species.nunique()   


# In[20]:


plt.scatter(ds.bill_length_mm,ds.flipper_length_mm)
plt.xlabel("bill_length_mm")
plt.ylabel("flipper_length_mm")
plt.show()


# In[21]:


x=ds.loc[:,["bill_length_mm","flipper_length_mm"]]
x


# In[22]:


from sklearn.cluster import KMeans


# In[23]:


km=KMeans(n_clusters=3)


# In[24]:


groups_content=km.fit_predict(x)
groups_content


# In[26]:


ds["groups_content"]=groups_content


# In[27]:


ds


# In[28]:


km.cluster_centers_         #centroides


# In[29]:


km.labels_


# In[30]:


km.n_iter_      #number of itirations 


# In[34]:


group1=ds[ds.groups_content==0]
group2=ds[ds.groups_content==1]
group3=ds[ds.groups_content==2]


# In[35]:


plt.scatter(group1.bill_length_mm,group1.flipper_length_mm,label="cluster = 1")
plt.scatter(group2.bill_length_mm,group2.flipper_length_mm,label="cluster = 2")
plt.scatter(group3.bill_length_mm,group3.flipper_length_mm,label="cluster = 3")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




