#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression Project 
# 
# In this project we will be working with a data set indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad
# 

# # Imports

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


sns.set_style('whitegrid')


# # Get the Data

# In[4]:


adv = pd.read_csv('advertising.csv')


# In[5]:


adv.head()


# # Variable Content

# In[9]:


adv.info()


# * 1) Daily Time Spent on Site --> numerical
# * 2) Age --> numerical
# * 3) Area Income --> numerical
# * 4) Daily Internet Usage --> numerical
# * 5) Male --> numerical
# * 6) Clicked on Ad --> numerical
# --------------------------------------
# * 7) Ad Topic Line --> object
# * 8) City --> object
# * 9) Country --> object
# * 10) Timestamp --> object

# In[7]:


# numeric variables overview
adv.describe()


# # Check Missing Data

# In[8]:


sns.heatmap(data = adv.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# # Exploratory Data Analysis - EDA

# ## Numerical Variables

# ### Daily Time Spent on Site vs Clicked on Ad

# In[13]:


plt.figure(figsize=(15,5))
sns.histplot(data = adv, x = 'Daily Time Spent on Site', hue = 'Clicked on Ad')


# * Data indicates strong inverse relationship between Daily Time Spent on Site and Clicked on Ad

# ### Age vs Clicked on Ad

# In[14]:


plt.figure(figsize=(15,5))
sns.histplot(data = adv, x = 'Age', hue = 'Clicked on Ad')


# * Data indicates some relationship between Age and Clicked on Ad

# ### Area Income vs Clicked on Ad

# In[15]:


plt.figure(figsize=(15,5))
sns.histplot(data = adv, x = 'Area Income', hue = 'Clicked on Ad')


# * Data indicates some relationship between Area Income and Clicked on Ad

# ### Daily Internet Usage vs Clicked on Ad

# In[16]:


plt.figure(figsize=(15,5))
sns.histplot(data = adv, x = 'Daily Internet Usage', hue = 'Clicked on Ad')


# * Data indicates strong inverse relationship between Daily Internet Usage and Clicked on Ad

# ### Male vs Clicked on Ad

# In[18]:


plt.figure(figsize=(15,5))
sns.countplot(data = adv, x = 'Male', hue = 'Clicked on Ad')


# * Data indicates low or no relationship between Male and Clicked on Ad

# ## Categorical Variables

# ### Ad Topic Line

# In[20]:


adv['Ad Topic Line'].value_counts()


# * The raw data presents no use for the model or apparent pattern for feature engineering

# ### City

# In[21]:


adv['City'].value_counts()


# * The raw data presents no use for the model or apparent pattern for feature engineering

# ### Country

# In[22]:


adv['Country'].value_counts()


# * The raw data presents no use for the model
# * Feature engineering --> possible categories through continents segmentation

# ### Timestamp

# In[23]:


adv['Timestamp'].value_counts()


# * The raw data presents no use for the model
# * Feature engineering --> possible categories through day of the week or parts of the day segmentation

# ## Variable Correlation

# In[27]:


plt.figure(figsize=(15,8))
sns.heatmap(data = adv.corr(), annot = True)


# Correlation Tier Hypothesis:
# * 1) Daily Internet Usage: -0.79
# * 2) Daily Time Spent on Site: -0,75
# * 3) Age: 0.49
# * 4) Area Income: -0.48
# * 5) Male: -0.038

# # Training as Testing Data

# In[28]:


adv.columns


# In[29]:


X = adv[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = adv['Clicked on Ad']


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# ## Training the Model

# In[32]:


from sklearn.linear_model import LogisticRegression


# In[33]:


model = LogisticRegression()


# In[34]:


model.fit(X_train, y_train)


# ## Predicting Test Data

# In[40]:


pred = model.predict(X_test)


# ## Evaluating the Model

# In[49]:


from sklearn.metrics import confusion_matrix, classification_report


# ### Test Size = 0.2

# In[48]:


print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))


# ### Test Size = 0.25

# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


# In[51]:


model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))


# ### Test Size = 0.3

# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[55]:


model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))


# ### Test Size = 0.35

# In[56]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=101)


# In[57]:


model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))


# ## Optimal Results

# ### Test Size = 0.3

# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[59]:


model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))


# * 93% Precision, Recall and F-1 Score indicates a very good fit for the model

# # Hypothesis: Drop Male Column to increase performance

# In[60]:


X = adv[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
y = adv['Clicked on Ad']


# In[61]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[62]:


model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))


# * No changes were observed
