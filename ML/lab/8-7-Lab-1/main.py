#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mistune.plugins.table import table_in_list
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# In[2]:


df = pd.read_csv("titanic.csv")

df.head()


# In[3]:


df.tail()


# In[4]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[5]:


corr = df.corr()
corr.Survived.sort_values(ascending=False).head(10)


# In[ ]:
