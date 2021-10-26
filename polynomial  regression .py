#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[2]:


ds=pd.read_csv("position_salaries.csv")
ds


# In[3]:


ds.isnull().sum()       


# In[4]:


x=ds.loc[:,["Level"]]


# In[5]:


y=ds.loc[:,["Salary"]]


# In[6]:


from sklearn.model_selection import train_test_split 


# In[7]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)


# In[8]:


plt.scatter(x,y)


# In[9]:


from sklearn.preprocessing import PolynomialFeatures


# In[37]:


polyf=PolynomialFeatures(degree =3)


# In[38]:


x_poly=polyf.fit_transform(x)


# In[39]:


from sklearn.linear_model import LinearRegression


# In[40]:


model=LinearRegression()


# In[41]:


model.fit(x_poly,y)


# In[42]:


y_poly_predict=model.predict(x_poly)
y_poly_predict


# In[43]:


plt.scatter(x,y)
plt.plot(x,y_poly_predict)
plt.show()                  #(degree=4 : overfitting) change it to 3


# In[ ]:





# In[ ]:





# In[ ]:




