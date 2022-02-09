#!/usr/bin/env python
# coding: utf-8

# # Unsupervised Learning - A K Means Clustering Project
# We will use a data frame with 777 observations on the following 18 variables from universities:
# 
# * Private: A factor with levels No and Yes indicating private or public university
# * Apps: Number of applications received
# * Accept: Number of applications accepted
# * Enroll: Number of new students enrolled
# * Top10perc: Pct. new students from top 10% of H.S. class
# * Top25perc: Pct. new students from top 25% of H.S. class
# * F.Undergrad: Number of fulltime undergraduates
# * P.Undergrad: Number of parttime undergraduates
# * Outstate: Out-of-state tuition
# * Room.Board: Room and board costs
# * Books: Estimated book costs
# * Personal: Estimated personal spending
# * PhD: Pct. of faculty with Ph.D.’s
# * Terminal: Pct. of faculty with terminal degree
# * S.F.Ratio: Student/faculty ratio
# * perc.alumni: Pct. alumni who donate
# * Expend: Instructional expenditure per student
# * Grad.Rate: Graduation rate

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


uni = pd.read_csv('College_Data')


# In[5]:


uni_original = uni.copy()


# In[6]:


uni.head()


# # Variable Content

# In[7]:


uni.info()


# In[8]:


uni.describe()


# # Check Missing or Duplicated Data

# In[9]:


sns.heatmap(data = uni.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# * No missing data

# In[10]:


uni.duplicated()


# * No duplicated data

# # Exploratory Data Analysis - EDA

# ## Response Variable - Private

# In[11]:


plt.figure(figsize=(15,5))
sns.countplot(data = uni, x = 'Private')


# In[12]:


uni['Private'].value_counts()


# ## Private: Categorical to Dummy Variable

# In[13]:


priv = pd.get_dummies(data = uni['Private'], drop_first = True, prefix = 'private')


# In[14]:


priv.head()


# ## Concatenating with Original Dataset

# In[15]:


uni = pd.concat([uni, priv], axis = 1)


# In[16]:


uni.head()


# ## Drop old Private Column

# In[17]:


uni.drop('Private', axis = 1, inplace = True)


# In[18]:


uni.head()


# ## Drop Name Column

# In[19]:


uni.drop('Name', axis = 1, inplace = True)


# In[20]:


uni.head()


# ## Variable Correlation

# In[21]:


plt.figure(figsize=(20,8))
sns.heatmap(data = uni.corr(), annot = True)


# Correlation Tier Hypothesis:
# * 1) F.Undergrad: -0.62
# * 2) Enroll: -0.57
# * 3) Outstate: 0.55
# * 4) Accept: -0.48
# * 5) S.F.Ratio: -0.47
# * 6) P.Undergrad: -0.45
# * 7) Apps: -0.43
# * 8) Per.alumni: 0.41
# * 9) Room.Board: 0.34
# * 10) Grad.Rate: 0.34
# * 11) Expend: 0.26

# ## F.Undergrad vs private_Yes

# In[22]:


plt.figure(figsize=(15,5))
sns.histplot(data = uni, x='F.Undergrad', hue = 'private_Yes')


# * Data indicates strong relationship between lower number of fulltime undergraduates and private universities

# ## Enroll vs private_Yes

# In[23]:


plt.figure(figsize=(15,5))
sns.histplot(data = uni, x='Enroll', hue = 'private_Yes')


# * Data indicates strong relationship with fewer new students enrolled and Private Universities

# ## Outstate vs private_Yes

# In[24]:


plt.figure(figsize=(15,5))
sns.histplot(data = uni, x='Outstate', hue = 'private_Yes')


# * Data indicates that Out of State Tuition Fees tend to be more expensive for Private Universities

# ## Accept vs private_Yes

# In[25]:


plt.figure(figsize=(15,5))
sns.histplot(data = uni, x='Accept', hue = 'private_Yes')


# * Data indicates that Private Universities accept fewer applications than Public Universities

# ## S.F.Ratio vs private_Yes

# In[26]:


plt.figure(figsize=(15,5))
sns.histplot(data = uni, x='S.F.Ratio', hue = 'private_Yes')


# * Data indicates that Private Universities have lower Student Faculty Ratios compared to Public Universities, meaning more teachers, fewer students, or both

# ## P.Undergrad vs private_Yes

# In[27]:


plt.figure(figsize=(15,5))
sns.histplot(data = uni, x='P.Undergrad', hue = 'private_Yes')


# * Data indicates that Private Universisties tend to have fewer parttime undergraduates

# ## Apps vs private_Yes

# In[28]:


plt.figure(figsize=(15,5))
sns.histplot(data = uni, x='Apps', hue = 'private_Yes')


# * Data indicates that Private Universities tend to receive fewer Applications than Public Universisites

# ## perc.alumni vs private_Yes

# In[29]:


plt.figure(figsize=(15,5))
sns.histplot(data = uni, x='perc.alumni', hue = 'private_Yes')


# * Data indicates that students enrolled in Private Universities tend to donate more percetwise

# ## Room.Board vs private_Yes

# In[30]:


plt.figure(figsize=(15,5))
sns.histplot(data = uni, x='Room.Board', hue = 'private_Yes')


# * Data indicates that Private Universities tend to have higher Room and Board Costs

# ## Grad.Rate vs private_Yes

# In[31]:


plt.figure(figsize=(15,5))
sns.histplot(data = uni, x='Grad.Rate', hue = 'private_Yes')


# * Data indicates that Private Universities tend to have higher Graduation Rates

# * 100 is the maximum Graduation Rate, meaning we must correct the data

# ## Graduation Rate > 100%

# In[32]:


uni['Grad.Rate'] = np.where(uni['Grad.Rate'] > 100, 100, uni['Grad.Rate'])


# In[33]:


plt.figure(figsize=(15,5))
sns.histplot(data = uni, x='Grad.Rate', hue = 'private_Yes')


# ## Expend vs private_Yes

# In[34]:


plt.figure(figsize=(15,5))
sns.histplot(data = uni, x='Expend', hue = 'private_Yes')


# * Data indicates that Instructional expenditure per student tends to be higher in Private Universities

# # Training the Model

# In[35]:


from sklearn.cluster import KMeans


# In[36]:


model = KMeans(n_clusters = 2)


# In[37]:


model.fit(uni.drop('private_Yes', axis = 1))


# In[38]:


model.cluster_centers_


# In[39]:


model.labels_


# ## Evaluating the Model

# Since we know which universities are public and which are private, we can evaluate our model!

# In[40]:


from sklearn.metrics import confusion_matrix, classification_report


# In[41]:


print(confusion_matrix(uni['private_Yes'], model.labels_))
print('\n')
print(classification_report(uni['private_Yes'], model.labels_))

