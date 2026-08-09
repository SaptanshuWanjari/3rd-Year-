#!/usr/bin/env python
# coding: utf-8

# # ML Lab 3
# ## Name: Saptanshu Wanjari
# ### Class: C1-A2
# ### Roll No: 18

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[2]:


df = pd.read_csv("USA_Housing.csv")
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
print(df.shape)


# ## EDA

# In[11]:


plt.figure(figsize=(8,5))
sns.histplot(df['Price'], kde=True)
plt.title('Distribution of House Price')
plt.show()


# In[12]:


plt.figure(figsize=(8,5))
sns.histplot(df['Avg. Area Income'], kde=True)
plt.title('Distribution of Avg. Area Income')
plt.show()


# In[13]:


sns.scatterplot(x='Avg. Area Income', y='Price', data=df)
plt.title('Avg. Area Income vs Price')
plt.show()


# In[14]:


plt.figure(figsize=(10,6))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()


# In[15]:


sns.boxplot(x=df['Avg. Area Income'])
plt.title('Box plot of Avg. Area Income')
plt.show()


# In[16]:


# IQR method to find outliers in Avg. Area Income
q1 = df['Avg. Area Income'].quantile(0.25)
q3 = df['Avg. Area Income'].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outliers = df[(df['Avg. Area Income'] < lower) | (df['Avg. Area Income'] > upper)]
print('Q1:', q1)
print('Q3:', q3)
print('IQR:', iqr)
print('Lower bound:', lower)
print('Upper bound:', upper)
print('Number of outliers:', len(outliers))


# ## Simple Linear Regression

# In[17]:


X = df[['Avg. Area Income']]
y = df['Price']


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Train shape:', X_train.shape)
print('Test shape:', X_test.shape)


# In[19]:


slr_model = LinearRegression()
slr_model.fit(X_train, y_train)


# In[20]:


y_pred = slr_model.predict(X_test)


# In[21]:


MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
R2 = r2_score(y_test, y_pred)

print('Simple Linear Regression Results')
print('--------------------------------')
print('MAE: ', MAE)
print('MSE: ', MSE)
print('RMSE: ', RMSE)
print('R2 Score:', R2)


# In[22]:


print('Intercept: ', slr_model.intercept_)
print('Slope: ', slr_model.coef_[0])


# In[23]:


plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Avg Area Income')
plt.ylabel('House Price')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()


# In[24]:


income = 80000
new_data = pd.DataFrame({"Avg. Area Income":[income]})
prediction = slr_model.predict(new_data)
print("Predicted House Price for income $80,000: $ {:,.2f}".format(prediction[0]))


# ## Hyperparameter Tuning with GridSearchCV

# In[25]:


lr = LinearRegression()
lr_params = {
    'fit_intercept': [True, False],
    'positive': [True, False]
}
grid_search = GridSearchCV(estimator=lr, param_grid=lr_params, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
print('Best Parameters: ', grid_search.best_params_)
print('Best Cross Validation Score: ', grid_search.best_score_)


# In[26]:


best_model = grid_search.best_estimator_
print(best_model)


# In[27]:


y_pred_best = best_model.predict(X_test)

MAE = mean_absolute_error(y_test, y_pred_best)
MSE = mean_squared_error(y_test, y_pred_best)
RMSE = np.sqrt(MSE)
R2 = r2_score(y_test, y_pred_best)

print('Best Model Results')
print('------------------')
print('MAE: ', MAE)
print('MSE: ', MSE)
print('RMSE: ', RMSE)
print('R2 Score:', R2)

