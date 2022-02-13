#!/usr/bin/env python
# coding: utf-8

# # Loan Payment Prediction - Lending Club
# The following dataset contains data on customers that got a loan through Lending Club, a p2p platform.
# 
# Here are what the columns represent:
# * credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
# * purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
# * int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
# * installment: The monthly installments owed by the borrower if the loan is funded.
# * log.annual.inc: The natural log of the self-reported annual income of the borrower.
# * dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
# * fico: The FICO credit score of the borrower.
# * days.with.cr.line: The number of days the borrower has had a credit line.
# * revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
# * revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
# * inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
# * delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
# * pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

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


lend = pd.read_csv('loan_data.csv')


# In[5]:


lend.head()


# # Variable Content

# In[6]:


lend.info()


# * All numerical varibles, but purpose

# In[7]:


lend.describe()


# # Check Missing or Duplicated Data

# In[8]:


sns.heatmap(data = lend.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# * No missing data

# In[9]:


lend.duplicated()


# * No duplicates

# # Exploratory Data Analysis

# ## Response Variable: Not Fully Paid

# In[10]:


sns.countplot(data = lend, x = 'not.fully.paid')


# In[11]:


lend['not.fully.paid'].value_counts()


# ## Credit Policy vs Not Fully Paid

# In[12]:


lend['credit.policy'].describe()


# In[13]:


plt.figure(figsize=(15,5))
sns.countplot(data = lend, x = 'credit.policy', hue = 'not.fully.paid')


# * Data indicates low or no relationship between meeting Credit Policy requirements and Not FullyPaid

# ## Purpose vs Not Fully Paid

# In[14]:


lend['purpose'].describe()


# In[15]:


plt.figure(figsize=(15,5))
sns.countplot(data = lend, x = 'purpose', hue = 'not.fully.paid')


# * Data indicates low o no relationship between Purpose and Not Fully Paid

# ## Interest Rate vs Not Fully Paid

# In[16]:


lend['int.rate'].describe()


# In[17]:


plt.figure(figsize=(15,5))
sns.histplot(data = lend, x = 'int.rate', hue = 'not.fully.paid')


# * Data indicates low or no relationship between Interest Rate and Not Fully Paid

# ## Installment vs Not Fully Paid

# In[18]:


lend['installment'].describe()


# In[19]:


plt.figure(figsize=(15,5))
sns.histplot(data = lend, x = 'installment', hue = 'not.fully.paid')


# * Data indicates low or no relationship between Installment and Not Fully Paid

# # Log Annual Income vs Not Fully Paid

# In[20]:


lend['log.annual.inc'].describe()


# In[21]:


plt.figure(figsize=(15,5))
sns.histplot(data = lend, x = 'log.annual.inc', hue = 'not.fully.paid')


# * Data indicates low or no relationship between Log Annual Income vs Not Fully Paid

# ## DTI vs Not Fully Paid

# In[22]:


lend['dti'].describe()


# In[23]:


plt.figure(figsize=(15,5))
sns.histplot(data = lend, x = 'dti', hue = 'not.fully.paid')


# * Data indicates low or no relationship between DTI and Not Fully Paid

# ## FICO vs Not Fully Paid

# In[24]:


lend['fico'].describe()


# In[25]:


plt.figure(figsize=(15,5))
sns.histplot(data = lend, x = 'fico', hue = 'not.fully.paid')


# * Data indicates low or no relationship between FICO and Not Fully Paid

# ## Days With Credit Line vs Not Fully Paid

# In[26]:


lend['days.with.cr.line'].describe()


# In[27]:


plt.figure(figsize=(15,5))
sns.histplot(data = lend, x = 'days.with.cr.line', hue = 'not.fully.paid')


# * Data indicates low or no relationship between Days With Credit Line and Not Fully Paid

# ## Revolving Balance vs Not Fully Paid

# In[28]:


lend['revol.bal'].describe()


# In[29]:


plt.figure(figsize=(20,8))
sns.histplot(data = lend, x = 'revol.bal', hue = 'not.fully.paid')


# * Data indicates low or no relationship between Revolving Balance and Not Fully Paid

# ## Revolving Line Utilization Rate vs Not Fully Paid

# In[30]:


lend['revol.util'].describe()


# In[31]:


plt.figure(figsize=(15,5))
sns.histplot(data = lend, x = 'revol.util', hue = 'not.fully.paid')


# * Data indicates low or no relationship between Relvolving Line Utilization Rate and Not Fully Paid

# ## Inquiries by Creditors Last 6 Months vs Not Fully Paid

# In[32]:


lend['inq.last.6mths'].describe()


# In[33]:


plt.figure(figsize=(15,5))
sns.countplot(data = lend[lend['inq.last.6mths'] <= 10], x = 'inq.last.6mths', hue = 'not.fully.paid')


# * Data indicates low or no relationship between Inquiries by Creditors Last 6 Months and Not Fully Paid

# ## Delinq 2 yrs (the number of times the borrower had been 30+ days past due on a payment in the past 2 years) vs Not Fully Paid

# In[34]:


lend['delinq.2yrs'].describe()


# In[35]:


plt.figure(figsize=(15,5))
sns.countplot(data = lend[lend['delinq.2yrs'] <= 3], x = 'delinq.2yrs', hue = 'not.fully.paid')


# * Data indicates low or no relationship between Delinq.2yrs and Not Fully Paid

# ## Variable Correlation

# In[36]:


plt.figure(figsize=(20,8))
sns.heatmap(data = lend.corr(), annot = True)


# ## Purpose: Categorical to Dummy Variable

# In[37]:


purp = pd.get_dummies(data = lend['purpose'], drop_first = True)


# In[38]:


purp.head()


# ## Concatenating with Original Dataset

# In[39]:


lend = pd.concat([lend, purp], axis = 1)


# In[40]:


lend.head()


# ## Drop Old Purpose Column

# In[41]:


lend.drop('purpose', axis = 1, inplace = True)


# In[42]:


lend.head()


# # Normalizing Varibles

# In[43]:


from sklearn.preprocessing import MinMaxScaler


# In[44]:


scaler = MinMaxScaler()


# In[45]:


scaler.fit(lend)


# In[46]:


scaled_features = scaler.transform(lend)


# In[47]:


lend_feat = pd.DataFrame(data = scaled_features, columns = lend.columns)


# In[48]:


lend_feat.head()


# # Training and Testing Data

# In[49]:


X = lend_feat.drop('not.fully.paid', axis = 1)
y = lend['not.fully.paid']


# In[50]:


from sklearn.model_selection import train_test_split


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# ## Training the Model

# In[52]:


from sklearn.ensemble import RandomForestClassifier


# In[54]:


model = RandomForestClassifier(n_estimators=100)


# In[55]:


model.fit(X_train, y_train)


# ## Predicting Test Data

# In[56]:


pred = model.predict(X_test)


# ## Evaluating the Model

# In[57]:


from sklearn.metrics import confusion_matrix, classification_report


# In[58]:


print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))

