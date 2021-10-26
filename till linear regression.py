#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns


# In[2]:


ds=pd.read_csv("indian_liver_patient.csv")
ds


# In[3]:


ds.isnull().sum()   #messing value task


# In[4]:


ds.columns[ds.isnull().any()]


# In[5]:


ds.interpolate(method ="linear",inplace=True)


# In[6]:


ds.isnull().sum()


# In[7]:


ds.info()


# In[8]:


ds.Gender.unique()   #catogorical data task


# In[9]:


ds.Gender.nunique()


# In[10]:


from sklearn.preprocessing import OneHotEncoder   #nominaldata


# In[11]:


OHE=OneHotEncoder(sparse=False)


# In[12]:


OHE_values=pd.DataFrame(OHE.fit_transform(ds[["Gender"]]))
OHE_values


# In[13]:


ds.drop("Gender", axis=1 ,inplace=True)


# In[14]:


ds


# In[17]:


new_ds=pd.concat([OHE_values,ds],axis=1)


# In[18]:


new_ds


# In[20]:


#Linear Regression task
x=new_ds.loc[:351 ,["Albumin"]]                       #independent variable
y=new_ds.loc[:351 ,["Albumin_and_Globulin_Ratio"]]   #dependent variable
x_test=new_ds.loc[352: ,["Albumin"]]
y_actual=new_ds.loc[352: ,["Albumin_and_Globulin_Ratio"]]


# In[23]:


plt.scatter(x,y)
plt.show()


# In[24]:


from sklearn.linear_model import LinearRegression


# In[25]:


Rmodel=LinearRegression()


# In[26]:


Rmodel.fit(x,y)       #training 


# In[27]:


y_predicted=Rmodel.predict(x_test)     #test
y_predicted


# In[28]:


plt.scatter(x_test,y_predicted)
plt.show()


# In[29]:


plt.scatter(x_test,y_actual)
plt.show()


# In[30]:


Rmodel.intercept_     #best point


# In[31]:


Rmodel.coef_          #sloope


# In[32]:


import sklearn.metrics as mc    #error ratio


# In[33]:


mc.mean_absolute_error (y_actual,y_predicted)


# In[ ]:




