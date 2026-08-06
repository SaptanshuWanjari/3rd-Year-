#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[2]:


df=pd.read_csv("USA_Housing.csv")
df.head()


# In[3]:


df.tail()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df.duplicated().sum()


# In[10]:


if 'Address' in df.columns:
	df.drop('Address', axis=1, inplace=True)


# In[11]:


plt.figure(figsize=(10,6))
corr_matrix=df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',fmt=".2f",linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()


# In[12]:


plt.figure(figsize=(8,5))
sns.histplot(df["Price"], kde=True)
plt.title("Distribution of House Price")
plt.show()


# In[13]:


plt.figure(figsize=(8,5))
sns.histplot(df["Avg. Area Income"], kde=True)
plt.title("Distribution of Avg. Area Income")
plt.show()


# In[14]:


sns.scatterplot(x="Avg. Area Income",y="Price",data=df)
plt.show()


# In[15]:


sns.boxplot(x=df["Avg. Area Income"])
plt.show()


# In[16]:


sns.boxplot(x=df["Avg. Area Income"],y=df["Price"])
plt.show()


# In[17]:


# IQR method to find outliers in Avg. Area Income
q1=df["Avg. Area Income"].quantile(0.25)
q3=df["Avg. Area Income"].quantile(0.75)
iqr=q3-q1
lower=q1-1.5*iqr
upper=q3+1.5*iqr
outliers=df[(df["Avg. Area Income"]<lower)|(df["Avg. Area Income"]>upper)]
print("Q1:",q1)
print("Q3:",q3)
print("IQR:",iqr)
print("Outlier rows in Avg. Area Income:",len(outliers))


# In[18]:


pd.set_option('display.float_format','{:.2f}'.format)


# In[19]:


df.head()


# ## SLR

# In[20]:


X=df[['Avg. Area Income']]
y=df['Price']


# In[21]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("Train shape:",X_train.shape)
print("Test shape:",X_test.shape)


# In[22]:


from sklearn.linear_model import LinearRegression

slr_model=LinearRegression()
slr_model.fit(X_train,y_train)


# In[23]:


y_pred=slr_model.predict(X_test)


# In[24]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print("Simple Linear Regression Results")
print("------------------------------------")

print("MAE: ",mean_absolute_error(y_test,y_pred))
print("MSE: ",mean_squared_error(y_test,y_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_test,y_pred)))
print("R2 Score:",r2_score(y_test,y_pred))


# In[25]:


print("Intercept: ",slr_model.intercept_)
print("Slope: ",slr_model.coef_[0])


# In[26]:


plt.figure(figsize=(8,6))
plt.scatter(X_test,y_test,color='blue',label="Actual Data")

plt.plot(X_test,y_pred,color='red',linewidth=2,label="Regression Line")

plt.xlabel("Avg Area Income")
plt.ylabel("House Price")
plt.title("Simple Linear Regression")
plt.legend()

plt.show()


# In[27]:


# predicting house price for a new income value
new_data = pd.DataFrame({"Avg. Area Income":[80000]})
prediction = slr_model.predict(new_data)
print("Predicted House Price for income 80000: ${:,.2f}".format(prediction[0]))

