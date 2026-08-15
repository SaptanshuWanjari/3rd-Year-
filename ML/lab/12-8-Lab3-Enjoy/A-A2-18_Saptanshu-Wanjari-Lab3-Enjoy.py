#!/usr/bin/env python
# coding: utf-8

# # ML Lab 3
# ## Name: Saptanshu Wanjari
# ### Class: A A2
# ### Roll No: 18

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("enjoy.csv")
print("Shape:", data.shape)
data.head()


# In[3]:


data.info()
print()
data.describe()


# In[4]:


print("Missing values per column:")
print(data.isnull().sum())
print("\nTotal missing:", data.isnull().sum().sum())


# In[5]:


dup = data.duplicated().sum()
print("Duplicate rows:", dup)
print("Total rows:", len(data))
print("Unique rows:", len(data) - dup)


# In[6]:


data_encoded = data.copy()
for col in data_encoded.columns:
    data_encoded[col] = data_encoded[col].astype('category').cat.codes
print("Encoded (0/1/2/... per category):")
data_encoded.head()


# In[7]:


fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(data.columns):
    data[col].value_counts().plot(kind='bar', ax=axes[i], color='skyblue')
    axes[i].set_title(col)
    axes[i].set_xlabel('')
plt.tight_layout()
plt.show()


# In[8]:


sns.countplot(x='EnjoySport', data=data, palette='Set2')
plt.title('EnjoySport target distribution')
plt.show()
print(data['EnjoySport'].value_counts())


# In[9]:


X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
print("X (features):\n", X)
print("\ny (target):\n", y)
print("\nX shape:", X.shape, "| y shape:", y.shape)


# In[10]:


def find_s(X, y):
    hypothesis = None
    print("Initial hypothesis:", hypothesis)
    for i in range(len(X)):
        if y[i] == "Yes":
            if hypothesis is None:
                hypothesis = X[i].copy()
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j] != X[i][j]:
                        hypothesis[j] = '?'
        print("\nAfter training example", i + 1, ":", hypothesis)
    return hypothesis

final_hypothesis = find_s(X, y)
print("\nFinal hypothesis:", final_hypothesis)


# In[11]:


def covers(g, h):
    return all(g[j] == "?" or g[j] == h[j] for j in range(len(g)))

def candidate_elimination(concepts, target):
    S = concepts[0].copy()
    G = [["?" for _ in range(len(S))]]
    print("Initial S:", S)
    print("Initial G:", G)
    for i, (h, t) in enumerate(zip(concepts, target)):
        if t == "Yes":
            for j in range(len(S)):
                if S[j] != h[j]:
                    S[j] = "?"
            G = [g for g in G if covers(g, h)]
        else:
            keep, spec = [], []
            for g in G:
                if covers(g, h):
                    for j in range(len(g)):
                        if g[j] == "?" and S[j] != "?":
                            s2 = list(g)
                            s2[j] = S[j]
                            if not covers(s2, h) and s2 not in spec:
                                spec.append(s2)
                else:
                    keep.append(g)
            G = keep + spec
            dedup, seen = [], set()
            for g in G:
                k = tuple(g)
                if k not in seen:
                    seen.add(k)
                    dedup.append(g)
            G = [g for g in dedup if not any(
                g != g2 and all(g2[j] == "?" or g2[j] == g[j] for j in range(len(g)))
                for g2 in dedup)]
        print("\nAfter example", i + 1, "(" + t + "):")
        print("S:", S)
        print("G:", G)
    return S, G

S, G = candidate_elimination(X, y)
print("\nFinal S:", S)
print("Final G:", G)


# ## Result
# 
# The Find-S algorithm gives a final hypothesis of `['Sunny', 'Warm', '?', '?', '?', '?']`, and Candidate Elimination converges S and G to the same single hypothesis, which means the version space holds exactly one consistent concept.
