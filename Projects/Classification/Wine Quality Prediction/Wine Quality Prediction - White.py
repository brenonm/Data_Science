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


# In[12]:


wine['quality'].value_counts()


# ## New Variable: Wine Quality

# In[13]:


white = wine.copy()


# In[14]:


white['wine_quality'] = np.where(white['quality'] > 6, 1, 0)


# In[15]:


white.head()


# In[16]:


sns.countplot(data = white, x = 'wine_quality')


# In[17]:


white.drop('quality', axis = 1, inplace = True)


# In[18]:


white.head()


# ### Fixed Acidity vs Wine Quality

# In[19]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'fixed acidity', hue = 'wine_quality')


# * Data indicates low or no relationship between Fixed Acidity and Wine Quality

# ### Volatile Acidity vs Wine Quality

# In[20]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'volatile acidity', hue = 'wine_quality')


# * Data indicates low or no relationship between Volatile Acidity and Wine Quality

# ### Citric Acid vs Wine Quality

# In[21]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'citric acid', hue = 'wine_quality')


# * Data indicates low or no relationship between Citric Acid and Wine Quality

# ### Residual Sugar vs Wine Quality

# In[22]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'residual sugar', hue = 'wine_quality')


# * Data indicates low or no relationship between Residual Sugar and Wine Quality

# ### Chlorides vs Wine Quality

# In[23]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'chlorides', hue = 'wine_quality')


# * Data indicates low inverse relationship between Chlorides and Wine Quality

# ### Free Sulfur Dioxide vs Wine Quality

# In[24]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'free sulfur dioxide', hue = 'wine_quality')


# * Data indicates low or no relationship between Free Sulfur Dioxie and Wine Quality

# ### Total Sulfur Dioxide vs Wine Quality

# In[25]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'total sulfur dioxide', hue = 'wine_quality')


# * Data indicates low inverse relationship between Total Sulfur Dioxide and Wine Quality

# ### Density vs Wine Quality

# In[26]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'density', hue = 'wine_quality')


# * Data indicates some inverse relationship between Density and Wine Quality

# ### pH vs Wine Quality

# In[27]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'pH', hue = 'wine_quality')


# * Data indicates low or no relationship between pH and Wine Quality

# ### Sulphates vs Wine Quality

# In[28]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'sulphates', hue = 'wine_quality')


# * Data indicates low or no relationship between Sulphates and Wine Quality

# ### Alcohol vs Wine Quality

# In[29]:


plt.figure(figsize=(15,5))
sns.histplot(data = white, x = 'alcohol', hue = 'wine_quality')


# * Data indicates some relationship between Alcohol and Wine Quality

# ## Variable Correlation

# In[30]:


plt.figure(figsize=(15,8))
sns.heatmap(data = white.corr(), annot = True)


# ### Alcohol vs Density

# In[31]:


plt.figure(figsize=(15,5))
sns.jointplot(data = white, x = 'alcohol', y = 'density', hue = 'wine_quality')


# # Normalizing Variables

# In[32]:


from sklearn.preprocessing import StandardScaler


# In[33]:


scaler = StandardScaler()


# In[34]:


scaler.fit(white.drop('wine_quality', axis = 1))


# In[35]:


scaled_features = scaler.transform(white.drop('wine_quality', axis = 1))


# In[36]:


white_feat = pd.DataFrame(data = scaled_features, columns = white.columns[:-1])


# In[37]:


white_feat.head()


# # Training and Testing Data

# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


X = white_feat
y = white['wine_quality']


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# ## Training the Model

# In[41]:


from sklearn.neighbors import KNeighborsClassifier


# ### k = 1

# In[42]:


model = KNeighborsClassifier(n_neighbors = 1)


# In[43]:


model.fit(X_train, y_train)


# ## Predicting Test Data

# In[44]:


pred = model.predict(X_test)


# ## Evaluating the Model

# In[45]:


from sklearn.metrics import confusion_matrix, classification_report


# In[46]:


print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))


# ## Apply Elbow Method to Discover Optimal k Value

# In[47]:


error_rate = []

for i in range(1,50):
    model = KNeighborsClassifier(n_neighbors = i )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    error_rate.append(np.mean(pred != y_test))


# In[48]:


plt.figure(figsize = (12,5))
plt.plot(range(1,50), error_rate, color='blue',
         linestyle = 'dashed',
         marker = 'o',
         markerfacecolor ='red',
         markersize = 10)

plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# ## GridSearchCV

# In[49]:


from sklearn.model_selection import GridSearchCV


# In[50]:



grid_params = {
    'n_neighbors':np.arange(1,52,2),
    'weights':['uniform', 'distance'],
    'metric':['euclidean', 'manhattan']   
}

gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv = 3, n_jobs = -1)
gs_results = gs.fit(X_train, y_train)


# In[51]:


gs_results.best_score_


# In[52]:


gs_results.best_estimator_


# In[53]:


gs_results.best_params_


# ## Optimal Results

# In[54]:


model = KNeighborsClassifier(metric='manhattan', n_neighbors=35, weights='distance')
model.fit(X_train, y_train)
pred = model.predict(X_test)


# In[55]:


print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))

