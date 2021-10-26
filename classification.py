#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
import seaborn as sns


# In[2]:


sns.get_dataset_names()


# In[4]:


ds=sns.load_dataset("iris")
ds


# In[5]:


ds.isnull().sum()


# In[8]:


ds.info()


# In[9]:


ds.species.unique()   #catogorical data task


# In[10]:


ds.species.nunique()


# In[11]:


from sklearn.preprocessing import OneHotEncoder   


# In[12]:


OHE=OneHotEncoder(sparse=False)


# In[13]:


OHE_values=pd.DataFrame(OHE.fit_transform(ds[["species"]]))
OHE_values


# In[14]:


ds.drop("species", axis=1 ,inplace=True)


# In[15]:


ds


# In[16]:


new_ds=pd.concat([OHE_values,ds],axis=1)


# In[24]:


new_ds


# In[25]:


x=new_ds.iloc[:,3:]
x


# In[26]:


y=new_ds.iloc[:,:3]
y


# In[27]:


from sklearn.model_selection  import train_test_split 


# In[28]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2)


# In[29]:


x_train


# In[30]:


from sklearn.neighbors import KNeighborsClassifier


# In[31]:


model=KNeighborsClassifier(n_neighbors=5)


# In[32]:


model.fit(x_train,y_train)       #training


# In[33]:


y_predict=model.predict(x_test)   #testing
y_predict


# In[35]:


import sklearn.metrics as mc


# In[37]:


mc.accuracy_score(y_test,y_predict)


# In[ ]:




