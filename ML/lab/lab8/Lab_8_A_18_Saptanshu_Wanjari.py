#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning Lab - 8
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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("ai4i2020.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.dtypes


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.isnull().sum()


# In[11]:


df.duplicated().sum()


# In[12]:


df["Type"].unique()


# In[13]:


df["Machine failure"].value_counts()


# In[14]:


numerical_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

df[numerical_features].hist(figsize=(15, 10), bins=30)
plt.show()


# In[15]:


plt.figure(figsize=(6,4))
sns.countplot(x="Machine failure", data=df)

plt.title("Distribution of Machine Failure")
plt.xlabel("Machine Failure")
plt.ylabel("Number of Machines")
plt.show()


# In[18]:


plt.figure(figsize=(12,8))

for i, column in enumerate(numerical_features,1):
    plt.subplot(2,3,i)
    sns.boxplot(y=df[column])
    plt.title(column)

plt.tight_layout()
plt.show()


# In[19]:


plt.figure(figsize=(12,7))
correlation = df[numerical_features + ["Machine failure"]].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# In[20]:


for column in numerical_features:
  plt.figure(figsize=(7,4))
  sns.boxplot(x="Machine failure", y=column, data=df)
  plt.title(f"{column} Vs Machine failure")
  plt.show()


# In[21]:


plt.figure(figsize=(7,5))
sns.countplot(x="Type",hue="Machine failure",data=df)
plt.title("Machine failure by product Type")
plt.xlabel("Product type")
plt.ylabel("Count")
plt.show()


# In[22]:


X= df[
    [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type"
]
]


# In[23]:


y=df["Machine failure"]


# In[24]:


print("Input shape",X.shape)
print("Output shape",y.shape)


# In[25]:


categorical_features =["Type"]
numerical_features =[
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# In[27]:


processor = ColumnTransformer(
    transformers=[
        ("num",StandardScaler(),numerical_features),
        ("cat",OneHotEncoder(handle_unknown='ignore'),categorical_features)
    ]
)


# In[28]:


linear_svm = Pipeline(
    steps=[("preprocessor", processor), ("classifier", SVC(kernel="linear"))]
)
linear_svm.fit(X_train, y_train)


# In[29]:


y_pred_linear = linear_svm.predict(X_test)


# In[30]:


accuracy_linear = accuracy_score(y_test, y_pred_linear)
precision_linear = precision_score(y_test, y_pred_linear)
recall_linear = recall_score(y_test, y_pred_linear)
f1_linear = f1_score(y_test, y_pred_linear)

print("Accuracy:", accuracy_linear)
print("Precision:", precision_linear)
print("Recall:", recall_linear)
print("F1 Score:", f1_linear)


# In[31]:


print(
    classification_report(
        y_test,
        y_pred_linear,
        target_names=["No Failure", "Failure"],
        zero_division=0
    )
)


# In[32]:


cm_linear = confusion_matrix(y_test, y_pred_linear)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_linear, annot=True, fmt="d", cmap="Set2",
            xticklabels=["No Failure","Failure"],
            yticklabels=["No Failure","Failure"]
            )
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Linear SVM")
plt.show()


# In[33]:


rbf_svm = Pipeline(
    steps=[("preprocessor", processor), ("classifier", SVC(kernel="rbf"))]
)
rbf_svm.fit(X_train, y_train)


# In[34]:


y_pred_rbf = rbf_svm.predict(X_test)


# In[35]:


accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
precision_rbf = precision_score(y_test, y_pred_rbf)
recall_rbf = recall_score(y_test, y_pred_rbf)
f1_rbf = f1_score(y_test, y_pred_rbf)

print("RBF SVM Results")
print("Accuracy:", accuracy_rbf)
print("Precision:", precision_rbf)
print("Recall:", recall_rbf)
print("F1 Score:", f1_rbf)


# In[36]:


print(classification_report(y_test,y_pred_rbf,target_names= ["No Failure","Failure"],zero_division=0))


# In[37]:


cm_rbf = confusion_matrix(y_test, y_pred_rbf)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_rbf, annot=True, fmt="d", cmap="Set2",
            xticklabels=["No Failure","Failure"],
            yticklabels=["No Failure","Failure"]
            )
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - RBF SVM")
plt.show()


# In[38]:


comparison =pd.DataFrame({
    "Model": ["Linear SVM", "RBF SVM"],
    "Accuracy": [accuracy_linear, accuracy_rbf],
    "Precision": [precision_linear, precision_rbf],
    "Recall": [recall_linear, recall_rbf],
    "F1 Score": [f1_linear, f1_rbf]
})
comparison


# In[39]:


comparison_plot = comparison.set_index("Model")

comparison_plot.plot(
    kind="bar",
    figsize=(10,6)
)
plt.title("Comparison of SVM Kernels")
plt.xlabel("SVM Kernel")
plt.ylabel("Score")
plt.ylim(0,1)
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


# In[41]:


param_grid = [
    {"classifier__kernel": ["linear"], "classifier__C": [0.1, 1, 10]},
    {"classifier__kernel": ["rbf"], "classifier__C": [1, 10, 50], "classifier__gamma": ["scale", 0.05, 0.1]}
]


# In[42]:


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=1,
    verbose=1
)


# In[43]:


grid_search.fit(X_train, y_train)
print("GridSearchCV completed successfully.")


# In[44]:


print("Best Parameters:")
print(grid_search.best_params_)


# In[45]:


print("Best Cross-Validation F1 Score:",grid_search.best_score_)


# In[46]:


best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)


# In[47]:


accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned, zero_division=0)
recall_tuned = recall_score(y_test, y_pred_tuned, zero_division=0)
f1_tuned = f1_score(y_test, y_pred_tuned, zero_division=0)

print("Tuned Model Results")
print("Accuracy:", accuracy_tuned)
print("Precision:", precision_tuned)
print("Recall:", recall_tuned)
print("F1 Score:", f1_tuned)


# In[48]:


final_comparison = pd.DataFrame({
    "Model": ["Linear SVM", "RBF SVM", "Tuned SVM"],
    "Accuracy": [accuracy_linear, accuracy_rbf, accuracy_tuned],
    "Precision": [precision_linear, precision_rbf, precision_tuned],
    "Recall": [recall_linear, recall_rbf, recall_tuned],
    "F1 Score": [f1_linear, f1_rbf, f1_tuned]
})
final_comparison


# In[49]:


final_comparison.set_index("Model").plot(kind="bar", figsize=(11, 6))
plt.title("Performance Comparison of Original and Tuned Models")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

