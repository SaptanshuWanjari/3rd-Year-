#!/usr/bin/env python
# coding: utf-8

# # ML Lab 4
# ## Name: Saptanshu Wanjari
# ### Class: A A2
# ### Roll No: 18
# 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("winequality-red.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


df.duplicated().sum()


# In[9]:


df=df.drop_duplicates()


# In[10]:


df.duplicated().sum()


# In[11]:


df.columns


# ## Univariate Analysis

# In[12]:


numeric_cols=df.columns

for col in numeric_cols:
	plt.figure(figsize=(7,4))
	sns.histplot(df[col],kde=True)
	plt.title(f"Distribution of {col}")
	plt.xlabel(col)
	plt.ylabel('Frequency')
	plt.show()


# In[13]:


for col in numeric_cols:
	plt.figure(figsize=(7,3))
	sns.boxplot(x=df[col])
	plt.title(f"Boxplot of {col}")
	plt.show()


# In[14]:


df.describe().T


# In[15]:


plt.figure(figsize=(7,4))
sns.countplot(x='quality',data=df)
plt.title("Distribution of Wine quality")
plt.show()


# In[16]:


df['quality'].value_counts().sort_index()


# ## Bivariate Analysis

# In[17]:


sns.boxplot(x='quality',y='alcohol',data=df)
plt.title('Alcohol vs Wine quality')
plt.show()


# In[18]:


df.head()


# In[19]:


features=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']

for col in features:
	plt.figure(figsize=(7,4))
	sns.boxplot(x='quality',y=col,data=df)
	plt.title(f"{col} vs Wine quality")
	plt.show()


# In[20]:


features=df.drop(columns=['quality'])
Q1=features.quantile(0.25)
Q3=features.quantile(0.75)

IQR=Q3-Q1

lower_bound = Q1-1.5*IQR
upper_bound = Q3+1.5*IQR

outlier_mask=(features<lower_bound) | (features>upper_bound)

outlier_count = outlier_mask.sum()

outlier_percentage=(outlier_count/len(features))*100

outlier_summary=pd.DataFrame({
	'Q1': Q1,
	'Q3': Q3,
	'IQR': IQR,
	'Lower Bound': lower_bound,
	'Upper Bound': upper_bound,
	'Outlier Count': outlier_count,
	'Outlier Percentage': outlier_percentage
})
outlier_summary.round(3)


# In[21]:


plt.figure(figsize=(12,6))

outlier_count.sort_values(ascending=False).plot(kind='bar')

plt.title("Number of outliers in each feature")
plt.xlabel("Features")
plt.ylabel("Number of outliers")
plt.xticks(rotation=45,ha='right')
plt.grid(axis='y',alpha=0.3)

plt.tight_layout()
plt.show()


# In[22]:


plt.figure(figsize=(7,4))
sns.countplot(x='quality',data=df)
plt.title("Distribution of wine quality")
plt.show()


# In[23]:


sns.scatterplot(x='alcohol',y='quality',data=df)
plt.title("Alcohol vs Wine quality")
plt.show()


# In[24]:


sns.scatterplot(x='volatile acidity',y='quality',data=df)
plt.title("Volatile acidity vs Wine quality")
plt.show()


# ## Multivariate Analysis

# In[25]:


plt.figure(figsize=(12,8))
corr=df.corr()
sns.heatmap(corr,annot=True,cmap='coolwarm',fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()


# In[26]:


selected=['alcohol','volatile acidity','sulphates','citric acid','quality']

sns.pairplot(df[selected],hue='quality')
plt.show()


# In[27]:


df['quality_group']=pd.cut(df['quality'],bins=[0,5,6,10],labels=['Low','Medium','High'])


# In[28]:


sns.boxplot(x='quality_group',y='alcohol',data=df)
plt.title("Alcohol distribution across quality groups")
plt.show()


# In[29]:


sns.countplot(x='quality_group',data=df)
plt.title("Count of wines in each quality group")
plt.show()


# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix,ConfusionMatrixDisplay)


# In[31]:


X=df.drop('quality',axis=1)
y=df['quality']


# In[32]:


print("X shape:",X.shape)
print("y shape:",y.shape)


# In[33]:


print(y.value_counts().sort_index())


# In[34]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


# In[35]:


criterion='entropy'


# In[36]:


dt_model=DecisionTreeClassifier(criterion=criterion,random_state=42)


# In[37]:


df['quality_group']=pd.cut(
	df['quality'],
	bins=[0,5,6,10],
	labels=['low','medium','high']
)


# In[38]:


print(X.columns)


# In[39]:


print(X.select_dtypes(exclude="number").columns)


# In[40]:


X=df.drop(columns=['quality','quality_group'],errors='ignore')


# In[41]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


# In[42]:


from sklearn.preprocessing import StandardScaler

# ponytail: Decision Trees are scale-invariant, scaling is for workflow completeness
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaling applied using StandardScaler")
print("Note: Decision Tree models are invariant to feature scaling")


# In[43]:


print(X.columns)
print(X.dtypes)


# In[44]:


print("X_train: ",X_train.shape)
print("X_test: ",X_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)


# In[45]:


dt_model.fit(X_train,y_train)


# In[46]:


dt_model.fit(X_train,y_train)


# In[47]:


y_train_pred=dt_model.predict(X_train)
y_test_pred=dt_model.predict(X_test)


# In[48]:


train_accuracy=accuracy_score(y_train,y_train_pred)
print("Training Accuracy: ",train_accuracy)


# In[49]:


test_accuracy=accuracy_score(y_test,y_test_pred)
print("Testing Accuracy: ",test_accuracy)


# In[50]:


print("Precision: ",precision_score(y_test,y_test_pred,average="weighted"))
print("Recall: ",recall_score(y_test,y_test_pred,average="weighted"))
print("F1 Score: ",f1_score(y_test,y_test_pred,average="weighted"))


# In[51]:


print(classification_report(y_test,y_test_pred))


# In[52]:


cm=confusion_matrix(y_test,y_test_pred)
plt.figure(figsize=(12,6))
sns.heatmap(cm,annot=True,cmap='coolwarm',fmt='d')
plt.xlabel('Predicted quality')
plt.ylabel('Actual quality')
plt.title("Confusion Matrix")
plt.show()


# In[53]:


print("tree depth: ",dt_model.get_depth())
print("number of leaves: ",dt_model.get_n_leaves())


# In[54]:


plt.figure(figsize=(25,15))
plot_tree(dt_model,feature_names=X.columns,class_names=[str(c) for c in sorted(y.unique())],filled=True,rounded=True,fontsize=8)
plt.title("Decision tree: Entropy")
plt.show()


# In[55]:


plt.figure(figsize=(24,12))
plot_tree(dt_model,feature_names=X.columns,class_names=[str(c) for c in sorted(y.unique())],filled=True,rounded=True,fontsize=8,impurity=True,precision=4,max_depth=3)
plt.title("Decision tree: Entropy")
plt.tight_layout()
plt.show()

