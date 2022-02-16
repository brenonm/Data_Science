#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Prediction

# Data Set: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
# 
# The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. For more details, consult: [Web Link] or the reference [Cortez et al., 2009]. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
# 
# These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent or poor wines. Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.
# 
# We will use Whine_Quality_White.
# 
# Input variables (based on physicochemical tests):
# 
#     1 - fixed acidity
#     2 - volatile acidity
#     3 - citric acid
#     4 - residual sugar
#     5 - chlorides
#     6 - free sulfur dioxide
#     7 - total sulfur dioxide
#     8 - density
#     9 - pH
#     10 - sulphates
#     11 - alcohol
# 
# Output variable (based on sensory data):
# 
#     12 - quality (score between 0 and 10)

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


wine = pd.read_csv('winequality-white.csv', sep = ';')


# In[5]:


wine.head()


# # Variable Content

# In[6]:


wine.info()


# * All numerical variables

# In[7]:


wine.describe()


# # Check Missing Data

# In[8]:


sns.heatmap(data = wine.isnull(),yticklabels = False, cbar = False, cmap = 'viridis' )


# * No missing data

# # Exploratory Data Analysis (EDA)

# In[9]:


wine.columns


# ## Response Variable: Quality

# In[10]:


plt.figure(figsize=(15,5))
sns.countplot(data = wine, x = 'quality', order = wine['quality'].value_counts().index)


# In[11]:


wine['quality'].value_counts()


# ## New Variable: Wine Quality

# In[12]:


white = wine.copy()


# In[13]:


white['wine_quality'] = np.where(white['quality'] > 6, 1, 0)


# In[14]:


white.head()


# In[15]:


sns.countplot(data = white, x = 'wine_quality')


# In[16]:


white.drop('quality', axis = 1, inplace = True)


# In[17]:


white.head()


# ### Fixed Acidity vs Wine Quality

# In[18]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'fixed acidity', hue = 'wine_quality')


# * Data indicates low or no relationship between Fixed Acidity and Wine Quality

# ### Volatile Acidity vs Wine Quality

# In[19]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'volatile acidity', hue = 'wine_quality')


# * Data indicates low or no relationship between Volatile Acidity and Wine Quality

# ### Citric Acid vs Wine Quality

# In[20]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'citric acid', hue = 'wine_quality')


# * Data indicates low or no relationship between Citric Acid and Wine Quality

# ### Residual Sugar vs Wine Quality

# In[21]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'residual sugar', hue = 'wine_quality')


# * Data indicates low or no relationship between Residual Sugar and Wine Quality

# ### Chlorides vs Wine Quality

# In[22]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'chlorides', hue = 'wine_quality')


# * Data indicates low inverse relationship between Chlorides and Wine Quality

# ### Free Sulfur Dioxide vs Wine Quality

# In[23]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'free sulfur dioxide', hue = 'wine_quality')


# * Data indicates low or no relationship between Free Sulfur Dioxie and Wine Quality

# ### Total Sulfur Dioxide vs Wine Quality

# In[24]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'total sulfur dioxide', hue = 'wine_quality')


# * Data indicates low inverse relationship between Total Sulfur Dioxide and Wine Quality

# ### Density vs Wine Quality

# In[25]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'density', hue = 'wine_quality')


# * Data indicates some inverse relationship between Density and Wine Quality

# ### pH vs Wine Quality

# In[26]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'pH', hue = 'wine_quality')


# * Data indicates low or no relationship between pH and Wine Quality

# ### Sulphates vs Wine Quality

# In[27]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'sulphates', hue = 'wine_quality')


# * Data indicates low or no relationship between Sulphates and Wine Quality

# ### Alcohol vs Wine Quality

# In[28]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'alcohol', hue = 'wine_quality')


# * Data indicates some relationship between Alcohol and Wine Quality

# ## Variable Correlation

# In[29]:


plt.figure(figsize=(15,8))
sns.heatmap(data = white.corr(), annot = True)


# ### Alcohol vs Density

# In[30]:


plt.figure(figsize=(15,5))
sns.jointplot(data = white, x = 'alcohol', y = 'density', hue = 'wine_quality')


# # Normalizing Variables

# In[31]:


from sklearn.preprocessing import StandardScaler


# In[32]:


scaler = StandardScaler()


# In[33]:


scaler.fit(white.drop('wine_quality', axis = 1))


# In[34]:


scaled_features = scaler.transform(white.drop('wine_quality', axis = 1))


# In[35]:


white_feat = pd.DataFrame(data = scaled_features, columns = white.columns[:-1])


# In[36]:


white_feat.head()


# # Training and Testing Data

# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


X = white_feat
y = white['wine_quality']


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# ## Training the Model

# In[40]:


from sklearn.neighbors import KNeighborsClassifier


# ### k = 1

# In[41]:


model = KNeighborsClassifier(n_neighbors = 1)


# In[42]:


model.fit(X_train, y_train)


# ## Predicting Test Data

# In[43]:


pred = model.predict(X_test)


# ## Evaluating the Model

# In[44]:


from sklearn.metrics import confusion_matrix, classification_report


# In[45]:


print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))


# ## Apply Elbow Method to Discover Optimal k Value

# ### Metric: Euclidean / Weights: Uniform

# In[66]:


error_rate = []

for i in range(1,50):
    model = KNeighborsClassifier(n_neighbors = i)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    error_rate.append(np.mean(pred != y_test))


# In[67]:


plt.figure(figsize = (12,5))
plt.plot(range(1,50), error_rate, color='blue',
         linestyle = 'dashed',
         marker = 'o',
         markerfacecolor ='red',
         markersize = 10)

plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[68]:


error_rate[9]


# ### Metric: Euclidean / Weights: Distance

# In[69]:


error_rate = []

for i in range(1,50):
    model = KNeighborsClassifier(n_neighbors = i, weights = 'distance')
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    error_rate.append(np.mean(pred != y_test))


# In[70]:


plt.figure(figsize = (12,5))
plt.plot(range(1,50), error_rate, color='blue',
         linestyle = 'dashed',
         marker = 'o',
         markerfacecolor ='red',
         markersize = 10)

plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[71]:


error_rate[12]


# ### Metric: Manhattan / Weights: Uniform

# In[72]:


error_rate = []

for i in range(1,50):
    model = KNeighborsClassifier(n_neighbors = i, metric = 'manhattan')
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    error_rate.append(np.mean(pred != y_test))


# In[73]:


plt.figure(figsize = (12,5))
plt.plot(range(1,50), error_rate, color='blue',
         linestyle = 'dashed',
         marker = 'o',
         markerfacecolor ='red',
         markersize = 10)

plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[74]:


error_rate[10]


# ### Metric: Manhattan / Weights: Distance

# In[75]:


error_rate = []

for i in range(1,50):
    model = KNeighborsClassifier(n_neighbors = i, metric= 'manhattan', weights= 'distance')
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    error_rate.append(np.mean(pred != y_test))


# In[76]:


plt.figure(figsize = (12,5))
plt.plot(range(1,50), error_rate, color='blue',
         linestyle = 'dashed',
         marker = 'o',
         markerfacecolor ='red',
         markersize = 10)

plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[78]:


error_rate[10]


# The following combination of paramaters presented the best performance:
# * __n_neighbours__ = 11
# * __metric__ = 'manhattan'
# * __weights__ = 'distance'
# 

# ## Optimal Results

# In[52]:


model = KNeighborsClassifier(metric='manhattan', n_neighbors=11, weights='distance')
model.fit(X_train, y_train)
pred = model.predict(X_test)


# In[53]:


print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))

