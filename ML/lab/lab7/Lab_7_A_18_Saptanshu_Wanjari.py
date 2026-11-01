#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning Lab - 7
# 
# ### Name : Saptanshu Wanjari
# 
# ### Class : A-A2
# 
# ### Roll no : 18

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,ConfusionMatrixDisplay,precision_score,recall_score,f1_score


# In[2]:


pd.set_option('display.max_columns',None)


# In[3]:


df=pd.read_excel('Raisin_Dataset.xlsx')


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df.dtypes


# In[10]:


df.duplicated().sum()


# In[11]:


df['Class'].unique()


# In[12]:


df['Class'].value_counts()


# **BAR PLOT**

# In[13]:


plt.figure(figsize=(10,5))
sns.countplot(x="Class",data=df)
plt.title("Distribution of Raisin Classes")
plt.xlabel("Raisin Class")
plt.ylabel("Number of Samples")
plt.show()


# **PIE-CHART**

# In[14]:


class_counts = df['Class'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Raisin Classes')
plt.show()


# In[15]:


numerical_features=df.select_dtypes(include=np.number).columns
print(numerical_features.tolist())


# **HISTOGRAM**

# In[16]:


df[numerical_features].hist(figsize=(15,10))
plt.suptitle("Distribution of Numerical Features",y=1.02,fontsize=16)
plt.tight_layout()
plt.show()


# **BOX PLOT**

# In[17]:


plt.figure(figsize=(15,10))
for i,column in enumerate(numerical_features):
    plt.subplot(3,3,i+1)
    sns.boxplot(y=df[column],color='skyblue')
    plt.tight_layout()
    plt.title(column)
plt.tight_layout()
plt.show()


# **HISTOGRAM WITH KDE FUNCTION**

# In[18]:


for column in numerical_features:
    plt.figure(figsize=(8,6))
    sns.histplot(data=df,x=column,hue="Class",kde=True,bins=25)
    plt.title(f"{column} Distribution by Class")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()


# **CORRELATION MATRIX**

# In[19]:


correlation_matrix=df[numerical_features].corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# In[20]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=df,x='Area',y='Perimeter',palette={'Kecimen':'Green','Besni':'blue'},hue='Class')
plt.title("Area vs Perimeter")
plt.xlabel("Area")
plt.ylabel("Perimeter")
plt.show()


# In[21]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=df,x='MajorAxisLength',y='MinorAxisLength',s=70,hue='Class')
plt.title("MajorAxisLength vs MinorAxislength")
plt.show()


# **DATA PREPROCESSING**

# In[22]:


label_encoder=LabelEncoder()
df['Class_encoded']=label_encoder.fit_transform(df['Class'])
print(df[['Class','Class_encoded']].head(10))


# In[23]:


print("Class Mapping:")
for index,class_name in enumerate(label_encoder.classes_):
    print(f"{index} : {class_name}")


# In[24]:


X=df[numerical_features]
y=df["Class_encoded"]
print("X shape",X.shape)
print("Y shape",y.shape)


# In[25]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Training Samples:",X_train.shape[0])
print("Testing Samples:",X_test.shape[0])


# In[26]:


GaussianNB()


# In[27]:


gnb=GaussianNB()
gnb.fit(X_train,y_train)


# In[28]:


y_pred=gnb.predict(X_test)
print(y_pred)


# In[29]:


comparison=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(comparison.head(10))


# In[30]:


comparison["Actual Class"]=label_encoder.inverse_transform(comparison["Actual"])
comparison["Predicted Class"]=label_encoder.inverse_transform(comparison["Predicted"])
print(comparison.head(10))


# In[31]:


accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
print("Accuracy Percentage:",accuracy*100,"%")


# In[32]:


precision=precision_score(y_test,y_pred,average='weighted')
print("Precision:",precision)
print("Precision Percentage:",precision*100,"%")


# In[33]:


recall=recall_score(y_test,y_pred,average='weighted')
print("Recall:",recall)
print("Recall Percentage:",recall*100,"%")


# In[34]:


f1=f1_score(y_test,y_pred,average='weighted')
print("F1 Score:",f1)
print("F1 Score Percentage:",f1*100,"%")


# In[35]:


print("=========MODEL PERFORMANCE=========")
print("Accuracy:",accuracy*100,"%")
print("Precision:",precision*100,"%")
print("Recall:",recall*100,"%")
print("F1 Score:",f1*100,"%")


# In[36]:


print("Classification Report:")
print(classification_report(y_test,y_pred,target_names=label_encoder.classes_))


# In[37]:


cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)


# In[38]:


plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=label_encoder.classes_,yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix - Gaussian Naive bayes")
plt.show()


# In[39]:


from sklearn.metrics import accuracy_score
y_train_pred=gnb.predict(X_train)
y_test_pred=gnb.predict(X_test)
test_accuracy=accuracy_score(y_test,y_test_pred)
print("Testing Accuracy:",test_accuracy)
print("Testing Accuracy Percentage:",test_accuracy*100,"%")
train_accuracy=accuracy_score(y_train,y_train_pred)
print("Training Accuracy:",train_accuracy)
print("Training Accuracy Percentage:",train_accuracy*100,"%")
#


# In[40]:


GaussianNB(priors=None)


# **MAIN_HYPERPARAMETERS**

# In[41]:


GaussianNB(var_smoothing=1e-9)


# In[42]:


print(gnb.get_params())


# In[43]:


smoothing_values=[1e-12,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]
result=[]
for var in smoothing_values:
    model=GaussianNB(var_smoothing=var)
    model.fit(X_train,y_train)
    prediction=model.predict(X_test)
    acc=accuracy_score(y_test,prediction)
    prec=precision_score(y_test,prediction)
    rec=recall_score(y_test,prediction)
    f1=f1_score(y_test,prediction)

    result.append([var,acc,prec,rec,f1])


# In[44]:


results_df=pd.DataFrame(result,columns=["var_smoothing","accuracy","precision","recall","f1"])
print(results_df)

