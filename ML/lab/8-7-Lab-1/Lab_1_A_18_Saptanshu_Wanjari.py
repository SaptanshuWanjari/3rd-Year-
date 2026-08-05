#!/usr/bin/env python
# coding: utf-8

# # Lab 1: Data Preprocessing on the Titanic Dataset
# 
# **Course:** Machine Learning &nbsp;|&nbsp; **Section:** A &nbsp;|&nbsp; **Roll No:** 18 &nbsp;|&nbsp; **Name:** Saptanshu Wanjari &nbsp;|&nbsp; **Batch:** A2
# 
# This notebook prepares the Titanic dataset for training a machine learning model by applying
# standard data preprocessing techniques.

# ## Aim
# To study and apply data preprocessing techniques on the Titanic dataset so that it becomes
# clean, consistent and ready for training a machine learning algorithm.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Step 1: Loading the Dataset
# We load the `titanic.csv` file into a pandas DataFrame and inspect its structure.

# In[2]:


df = pd.read_csv("titanic.csv")
print("Dataset shape:", df.shape)
df.head()


# In[3]:


df.tail()


# In[4]:


df.info()


# In[5]:


df.describe()


# ## Step 2: Handling Missing Values
# First we check how many missing values each column contains. The `Age` and `Embarked` columns
# have null entries. `Cabin` is mostly missing, so it will be dropped later.

# In[6]:


df.isnull().sum()


# In[7]:


df["Embarked"].unique()


# In[8]:


df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
print(df["Embarked"].value_counts())


# In[9]:


df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
print("Missing values after filling:")
df.isnull().sum()


# ## Step 3: Removing Irrelevant Columns
# Columns that carry no predictive value (IDs, names, ticket numbers) or that are mostly
# missing (`Cabin`) are removed from the dataset.

# In[10]:


df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
print("Shape after dropping columns:", df.shape)


# ## Step 4: Handling Duplicate Rows
# Duplicate rows add no information and can bias the model, so they are detected and removed.

# In[11]:


print("Duplicate rows before removal:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Shape after removing duplicates:", df.shape)


# ## Step 5: Detecting and Removing Outliers
# Box plots are used to detect outliers in the `Age` and `Fare` columns, which are then removed
# using the Inter-Quartile Range (IQR) rule.

# In[12]:


fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(x=df["Age"], ax=axes[0]).set_title("Box Plot of Age")
sns.boxplot(x=df["Fare"], ax=axes[1]).set_title("Box Plot of Fare")
plt.tight_layout()
plt.show()


# In[13]:


for col in ["Age", "Fare"]:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df = df[(df[col] > lower) & (df[col] < upper)]
print("Shape after removing outliers:", df.shape)


# In[14]:


fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(x=df["Age"], ax=axes[0]).set_title("Age after outlier removal")
sns.boxplot(x=df["Fare"], ax=axes[1]).set_title("Fare after outlier removal")
plt.tight_layout()
plt.show()


# ## Step 6: Encoding Categorical Features
# Machine learning algorithms work with numbers, so the text columns `Sex` and `Embarked`
# are converted into numeric form using `LabelEncoder`.

# In[15]:


le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
print("Data types after encoding:")
df.dtypes


# In[16]:


df.head()


# ## Step 7: Exploratory Data Analysis
# Visualising the data helps us understand the class distribution and the relationship
# between the features and the target variable `Survived`.

# In[17]:


sns.countplot(x="Survived", data=df)
plt.title("Count of Survived vs Not Survived")
plt.show()


# In[18]:


sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.show()


# In[19]:


sns.histplot(df["Age"], kde=True)
plt.title("Distribution of Age")
plt.show()


# In[20]:


sns.histplot(df["Fare"], kde=True)
plt.title("Distribution of Fare")
plt.show()


# In[21]:


sns.scatterplot(x="Age", y="Fare", data=df)
plt.title("Age vs Fare")
plt.show()


# In[22]:


plt.figure(figsize=(9, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# ## Step 8: Feature Scaling
# `Age` and `Fare` are measured on very different scales. Standardization gives both features
# a mean of 0 and a standard deviation of 1.

# In[23]:


scaler = StandardScaler()
df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])
print("Mean of Age after scaling:", round(df["Age"].mean(), 4))
print("Std of Age after scaling:", round(df["Age"].std(), 4))
print("Mean of Fare after scaling:", round(df["Fare"].mean(), 4))
print("Std of Fare after scaling:", round(df["Fare"].std(), 4))


# In[24]:


df.head()


# ## Step 9: Train-Test Split
# The cleaned dataset is split into training and testing sets. The model will be trained on
# 80% of the data and evaluated on the remaining 20%.

# In[25]:


X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ## Conclusion
# The Titanic dataset has been successfully preprocessed:
# 
# - Missing values in `Age` (filled with median) and `Embarked` (filled with mode) were handled.
# - Irrelevant columns (`PassengerId`, `Name`, `Ticket`, `Cabin`) and duplicate rows were removed.
# - Outliers in `Age` and `Fare` were removed using the IQR rule.
# - Categorical columns `Sex` and `Embarked` were label-encoded into numbers.
# - `Age` and `Fare` were standardized using `StandardScaler`.
# - The data was split into training (80%) and testing (20%) sets.
# 
# The resulting dataset is now clean, numeric and scaled, and is ready for training a machine
# learning model.
