#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation
# Congratulations! You just got some contract work with an Ecommerce company based in New York City that sells clothing online but they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want.
# 
# The company would like to know if they can segment their customers in groups with different spending habits.
# 
# We'll work with the Ecommerce Customers csv file from the company. It has Customer info, suchas __Email__, __Address__, and their color __Avatar__. Then it also has numerical value columns:
# 
# * __Avg. Session Length__: Average session of in-store style advice sessions.
# * __Time on App__: Average time spent on App in minutes
# * __Time on Website__: Average time spent on Website in minutes
# * __Length of Membership__: How many years the customer has been a member. 

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


customers = pd.read_csv('Ecommerce Customers')


# In[5]:


customers.head()


# In[6]:


customers_original = customers.copy()


# # Variable Information

# In[8]:


customers.info()


# In[9]:


customers.describe()


# # Check Missing or Duplicated Data

# In[10]:


sns.heatmap(data = customers.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# * No missing data

# In[12]:


customers.duplicated().value_counts()


# * No duplicated data

# # Exploratory Data Analysis - EDA

# ## Email

# In[14]:


customers['Email'].value_counts()


# * Varible will need feature engineering to generate insights

# In[16]:


customers.drop('Email', axis = 1, inplace = True)


# ## Address

# In[19]:


customers['Address'].value_counts()


# * Varible will need feature engineering to generate insights

# In[20]:


customers.drop('Address', axis = 1, inplace = True)


# ## Avatar

# In[21]:


customers['Avatar'].value_counts()


# * Varible will need feature engineering to generate insights

# In[22]:


customers.drop('Avatar', axis = 1 , inplace = True)


# ## Variable Correlation

# In[24]:


plt.figure(figsize=(20,8))
sns.heatmap(data = customers.corr(), annot = True)


# # Normalizing Variables

# In[31]:


from sklearn.preprocessing import MinMaxScaler


# In[32]:


scaler = MinMaxScaler()


# In[33]:


scaler.fit(customers)


# In[34]:


scaled_features = scaler.transform(customers)


# In[35]:


customers_feat = pd.DataFrame(data = scaled_features, columns = customers.columns)


# In[36]:


customers_feat.head()


# # Training the Model

# In[25]:


from sklearn.cluster import KMeans


# ## Clusters = 2

# In[37]:


model = KMeans(n_clusters = 2)


# In[38]:


model.fit(customers_feat)


# In[39]:


model.cluster_centers_


# In[40]:


model.labels_


# In[41]:


model.inertia_


# ## Apply Elbow Method por Optimal Number of Clusters

# In[47]:


inert = []
for i in range(1, 11):
    model = KMeans(n_clusters = i)
    model.fit(customers_feat)
    inert.append(model.inertia_)


# In[48]:


plt.figure(figsize=(15,5))
plt.plot(range(1, 11), inert)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[50]:


inert


# ## Visualizing the Clusters

# In[58]:


model = KMeans(n_clusters = 2)


# In[59]:


customers_feat['Cluster'] = model.fit_predict(customers_feat)
customers_feat['Cluster'] = customers_feat['Cluster'].astype('category')


# In[62]:


customers_feat.head()


# In[66]:


sns.catplot(data = customers_feat, x = 'Yearly Amount Spent', y = 'Cluster', kind = 'boxen')

