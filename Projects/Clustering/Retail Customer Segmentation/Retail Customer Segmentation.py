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

# In[7]:


customers.info()


# In[8]:


customers.describe()


# # Check Missing or Duplicated Data

# In[9]:


sns.heatmap(data = customers.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# * No missing data

# In[10]:


customers.duplicated().value_counts()


# * No duplicated data

# # Exploratory Data Analysis - EDA

# ## Email

# In[11]:


customers['Email'].value_counts()


# * Varible will need feature engineering to generate insights

# In[12]:


customers.drop('Email', axis = 1, inplace = True)


# ## Address

# In[13]:


customers['Address'].value_counts()


# * Varible will need feature engineering to generate insights

# In[14]:


customers.drop('Address', axis = 1, inplace = True)


# ## Avatar

# In[15]:


customers['Avatar'].value_counts()


# * Varible will need feature engineering to generate insights

# In[16]:


customers.drop('Avatar', axis = 1 , inplace = True)


# ## Variable Correlation

# In[17]:


plt.figure(figsize=(20,8))
sns.heatmap(data = customers.corr(), annot = True)


# # Normalizing Variables

# In[18]:


from sklearn.preprocessing import MinMaxScaler


# In[19]:


scaler = MinMaxScaler()


# In[20]:


scaler.fit(customers)


# In[21]:


scaled_features = scaler.transform(customers)


# In[22]:


customers_feat = pd.DataFrame(data = scaled_features, columns = customers.columns)


# In[23]:


customers_feat.head()


# # Training the Model

# In[24]:


from sklearn.cluster import KMeans


# ## Clusters = 2

# In[25]:


model = KMeans(n_clusters = 2)


# In[26]:


model.fit(customers_feat)


# In[27]:


model.cluster_centers_


# In[28]:


model.labels_


# In[29]:


model.inertia_


# ## Apply Elbow Method por Optimal Number of Clusters

# In[30]:


inert = []
for i in range(1, 11):
    model = KMeans(n_clusters = i)
    model.fit(customers_feat)
    inert.append(model.inertia_)


# In[31]:


plt.figure(figsize=(15,5))
plt.plot(range(1, 11), inert)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[32]:


inert


# ## Visualizing the Clusters

# In[33]:


model = KMeans(n_clusters = 2)


# In[34]:


customers_feat['Cluster'] = model.fit_predict(customers_feat)
customers_feat['Cluster'] = customers_feat['Cluster'].astype('category')


# In[35]:


customers_feat.head()


# ## Clusters by Features

# In[59]:


fig, axes = plt.subplots(2,3, figsize = (18,10))

fig.suptitle('Clusters by Features', fontsize = 20)

sns.boxenplot(data = customers_feat,
            x = 'Avg. Session Length',
            y = 'Cluster',
            ax = axes[0,0])

sns.boxenplot(data = customers_feat,
            x = 'Time on App',
            y = 'Cluster',
            ax = axes[0,1])

sns.boxenplot(data = customers_feat,
            x = 'Time on Website',
            y = 'Cluster',
            ax = axes[0,2])

sns.boxenplot(data = customers_feat,
            x = 'Length of Membership',
            y = 'Cluster',
            ax = axes[1,0])

sns.boxenplot(data = customers_feat,
            x = 'Yearly Amount Spent',
            y = 'Cluster',
            ax = axes[1,1])


# ## Yearly Amount Spent vs Cluster

# In[42]:


sns.catplot(data = customers_feat,
            x = 'Yearly Amount Spent',
            y = 'Cluster',
            kind = 'boxen')
plt.title('Yearly Amount Spent vs Cluster', fontsize = 20)


# ## Time on App vs Cluster

# In[45]:


sns.catplot(data = customers_feat,
            x = 'Time on App',
            y = 'Cluster',
            kind = 'boxen')
plt.title('Time on App vs Cluster', fontsize = 20)


# ## Time on Website vs Cluster

# In[54]:


sns.boxenplot(data = customers_feat,
            x = 'Time on Website',
            y = 'Cluster'
             )
plt.title('Time on Website vs Cluster', fontsize = 20)


# ## Conclusions 💯
# * The data __indicates__ that the company could develop different strategies for __customer targeting__ based on __Yearly Amount Spent__, such as:
#  
#  1) Focus on customers who spend __less__:  
#     
#     1.1) __Free shipping__  
#     1.2) __Progressive discounts__  
#     1.3) __Increase payment options__  
#  
#  2) Focus on customers who spend __more__:
#  
#     2.1) __Offer similar products through email marketing__  
#     2.2) __Offer in-cart complementary products__ (E.g: customer adds a shirt, receives recommendation for a pair of pants)  
#     2.3) __Develop Loyalty Program to maintain/increase high value customers' engagement__
