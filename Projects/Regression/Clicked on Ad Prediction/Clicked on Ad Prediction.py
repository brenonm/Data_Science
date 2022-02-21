#!/usr/bin/env python
# coding: utf-8

# # To Click or Not to Click - An Advertisement Prediction
# 
# In this project we will be working with a data set indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# 
# * __Daily Time Spent on Site__: consumer time on site in minutes
# * __Age__: cutomer age in years
# * __Area Income__: Avg. Income of geographical area of consumer
# * __Daily Internet Usage__: Avg. minutes a day consumer is on the internet
# * __Ad Topic Line__: Headline of the advertisement
# * __City__: City of consumer
# * __Male__: Whether or not consumer was male
# * __Country__: Country of consumer
# * __Timestamp__: Time at which consumer clicked on Ad or closed window
# * __Clicked on Ad__: 0 or 1 indicated clicking on Ad
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

# In[6]:


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


# # Check for Missing or Duplicated Data

# In[8]:


sns.heatmap(data = adv.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# * No missing data

# In[11]:


adv.duplicated().value_counts()


# * No duplicated data

# # Exploratory Data Analysis - EDA

# ## Numerical Variables

# ### Daily Time Spent on Site vs Clicked on Ad

# In[12]:


plt.figure(figsize=(15,5))
sns.histplot(data = adv, x = 'Daily Time Spent on Site', hue = 'Clicked on Ad')


# * Data indicates strong inverse relationship between Daily Time Spent on Site and Clicked on Ad

# ### Age vs Clicked on Ad

# In[13]:


plt.figure(figsize=(15,5))
sns.histplot(data = adv, x = 'Age', hue = 'Clicked on Ad')


# * Data indicates some relationship between Age and Clicked on Ad

# ### Area Income vs Clicked on Ad

# In[14]:


plt.figure(figsize=(15,5))
sns.histplot(data = adv, x = 'Area Income', hue = 'Clicked on Ad')


# * Data indicates some relationship between Area Income and Clicked on Ad

# ### Daily Internet Usage vs Clicked on Ad

# In[15]:


plt.figure(figsize=(15,5))
sns.histplot(data = adv, x = 'Daily Internet Usage', hue = 'Clicked on Ad')


# * Data indicates strong inverse relationship between Daily Internet Usage and Clicked on Ad

# ### Male vs Clicked on Ad

# In[16]:


plt.figure(figsize=(15,5))
sns.countplot(data = adv, x = 'Male', hue = 'Clicked on Ad')


# * Data indicates low or no relationship between Male and Clicked on Ad

# ## Categorical Variables

# ### Ad Topic Line

# In[17]:


adv['Ad Topic Line'].value_counts()


# * The raw data presents no use for the model or apparent pattern for feature engineering

# ### City

# In[18]:


adv['City'].value_counts()


# * The raw data presents no use for the model or apparent pattern for feature engineering

# ### Country

# In[19]:


adv['Country'].value_counts()


# * The raw data presents no use for the model
# * Feature engineering --> possible categories through continents segmentation

# ### Timestamp

# In[20]:


adv['Timestamp'].value_counts()


# * The raw data presents no use for the model
# * Feature engineering --> possible categories through day of the week or parts of the day segmentation

# ## Variable Correlation

# In[21]:


plt.figure(figsize=(15,8))
sns.heatmap(data = adv.corr(), annot = True)


# Correlation Tier Hypothesis:
# * 1) Daily Internet Usage: -0.79
# * 2) Daily Time Spent on Site: -0.75
# * 3) Age: 0.49
# * 4) Area Income: -0.48
# * 5) Male: -0.038

# # Normalizing Variables

# In[22]:


from sklearn.preprocessing import StandardScaler


# In[23]:


scaler = StandardScaler()


# In[24]:


scaler.fit(adv[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']])


# In[25]:


scaled_features = scaler.transform(adv[['Daily Time Spent on Site', 'Age', 'Area Income',
                                        'Daily Internet Usage', 'Male']])


# In[26]:


adv_feat = pd.DataFrame(data = scaled_features, columns = [['Daily Time Spent on Site', 'Age', 'Area Income',
                                        'Daily Internet Usage', 'Male']])


# In[27]:


adv_feat.head()


# # Training and Testing Data

# In[28]:


X = adv_feat
y = adv['Clicked on Ad']


# In[29]:


from sklearn.model_selection import train_test_split


# In[97]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# ## Training the Model

# In[98]:


from sklearn.linear_model import LogisticRegression


# In[99]:


model = LogisticRegression()


# In[100]:


model.fit(X_train, y_train)


# ## Predicting Test Data

# In[89]:


pred = model.predict(X_test)


# ## Evaluating the Model

# ### Test Size = 0.2

# In[35]:


from sklearn.metrics import confusion_matrix, classification_report


# In[90]:


print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))


# ### ROC Curve

# In[96]:


from sklearn.metrics import roc_curve


# In[ ]:


from sklearn.metrics import roc_auc_score


# In[95]:


fpr, tpr, thresholds = roc_curve(y_test, pred, pos_label = True)

plt.figure(figsize=(15,8))

plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12
plt.title('ROC curve for Clicked on Ad Classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

ROC_AUC = roc_auc_score(y_test, pred)

print('ROC AUC : {:.4f}'.format(ROC_AUC))


# ### Test Size = 0.3

# In[101]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[102]:


model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))


# ### ROC Curve

# In[104]:


Rfpr, tpr, thresholds = roc_curve(y_test, pred, pos_label = True)

plt.figure(figsize=(15,8))

plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12
plt.title('ROC curve for Clicked on Ad Classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

ROC_AUC = roc_auc_score(y_test, pred)

print('ROC AUC : {:.4f}'.format(ROC_AUC))


# ## Conclusions 💯
# * The data __indicates__ that a __Logistic Regression Model__ is an __excellent__ predictor of whether or not a customer would click on this company's Ad.
# 
# 
# * The data __also__ indicates the following relationships:  
#   
#   1) The __greater__ the Internet Usage, the __lower__ the odds of clicking on the Ad  
#   2) The __greater__ the Time Spent on Site, the __lower__ the odds of clicking on the Ad  
#   3) The __greater__ the Area Income, the __lower__ the odds of clicking on the Ad  
#   4) The __greater__ the Age, the __higher__ the odds of cliking on the Ad  
#   5) __Gender__ has __little or no relationship__ with clicking on the add or not  
# 
# 
# * If the __company's goal__ was to target __older__, __low area income__ and __both genders__ of customers, the Ad seems to __perform well__.
# 
# 
# * __However__, if the company's goal was to target any of these __specific groups__ or __any combination__ between them:
# 
#   1) Specifically __younger customers__  
#   2) Specifically __Men__  
#   3) Specifically __Women__  
#   4) Specifically __higher income customers__
# 
# 
# * The Ad __does not seem__ to convert the right __type__ of customers.
