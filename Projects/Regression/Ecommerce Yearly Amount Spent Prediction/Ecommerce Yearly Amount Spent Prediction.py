#!/usr/bin/env python
# coding: utf-8

# # Ecommerce Data Project
# 
# Congratulations! You just got some contract work with an Ecommerce company based in New York City that sells clothing online but they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want.
# 
# The company is trying to decide whether to focus their efforts on their mobile app experience or their website. They've hired you on contract to help them figure it out! Let's get started!
# 
# We'll work with the Ecommerce Customers csv file from the company. It has Customer info, such as Email, Address, and their color Avatar. Then it also has numerical value columns:
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


# # Variable Content

# In[6]:


customers.info()


# In[7]:


# numeric variables overview
customers.describe()


# # Check Missing Data

# In[8]:


sns.heatmap(data = customers.isnull(), yticklabels = False, cbar = False, cmap = 'viridis' )


# # Exploratory Data Analysis

# In[9]:


customers.columns


# ## Avg. Session Length vs Yearly Amount Spent

# In[10]:


sns.jointplot(data = customers, x = 'Avg. Session Length', y = 'Yearly Amount Spent')


# * Avg. Session Length similar to normal distribution
# * Yearly Amount Spent similar to normal distribuition
# * Avg. Session Lenght vs Yearly Amount Spent --> some correlation

# ## Time on App vs Yearly Amount Spent

# In[11]:


sns.jointplot(data = customers, x = 'Time on App', y = 'Yearly Amount Spent')


# * Time on App similar do normal distribution
# * Time on App vs Yearly Amount Spent --> strong correlation

# ## Time on Website vs Yearly Amount Spent

# In[12]:


sns.jointplot(data = customers, x = 'Time on Website', y = 'Yearly Amount Spent')


# * Time on Webciste similar to normal distribution
# * Time on Website vs Yearly Amount Spent --> low or no correlation

# ## Length of Membership vs Yearly Amount Spent

# In[36]:


g = sns.jointplot(data = customers, x = 'Length of Membership', y = 'Yearly Amount Spent')
g.fig.suptitle('Length of Membership (years) x Yearly Amount Spent ($)', weight = 'bold', fontsize = 15)
g.fig.subplots_adjust(top=0.90)


# * Length of Membership similar to normal distribution
# * Length of Membership vs Yearly Amount Spent --> very strong correlation

# ## Variable Correlation

# In[20]:


plt.figure(figsize=(15,5))
sns.heatmap(data = customers.corr(), annot = True)


# Correlation Tier Hypothesis:
# * 1) Length of Membership: 0.81
# * 2) Time on App: 0.5
# * 3) Avg. Session Length: 0.36
# * 4) Time on Website: -0.0026

# # Training as Testing Data

# In[21]:


customers.columns


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# ## Training the Model

# In[25]:


from sklearn.linear_model import LinearRegression


# In[26]:


model = LinearRegression()


# In[27]:


model.fit(X_train, y_train)


# In[28]:


# coefficients
coefficients = pd.DataFrame(data = model.coef_, index = X.columns, columns = ['Values'])


# In[31]:


coefficients['Values'].sort_values(ascending = False)


# ## Predicting Test Data

# In[33]:


pred = model.predict(X_test)


# In[34]:


# scatter plot of real vs predicted values --> the closer to a line, the better
plt.scatter(y_test, pred)
plt.xlabel('Y Test (True Values)')
plt.ylabel('Predictions')


# ## Evaluating the Model

# In[38]:


from sklearn import metrics


# ### MAE --> Mean Absolute Error

# In[36]:


mae = metrics.mean_absolute_error(y_test, pred)


# In[37]:


mae


# ### MSE --> Mean Squared Error

# In[40]:


mse = metrics.mean_squared_error(y_test, pred)


# In[41]:


mse


# ### RMSE --> Root Mean Squared Error

# In[43]:


rmse = metrics.mean_squared_error(y_test, pred, squared = False)


# In[44]:


rmse


# ### R²

# In[45]:


r2 = metrics.r2_score(y_test, pred)


# In[46]:


r2


# In[47]:


indicators = pd.array([mae, mse, rmse, r2])


# In[48]:


results = pd.DataFrame(data = indicators, index = ['MAE','MSE','RMSE','R²'], columns = ['Values'])


# In[50]:


results


# * 98,81% fit indicates a very good fit for the model

# # Coefficient Interpretarion

# In[51]:


coefficients


# * 1) A 1 unit increase in Length of Membership increases 61.47 units in Yearly Amount Spent --> most important
# * 2) A 1 unit increase in Time on App increases 38.71 units in Yearly Amount Spent --> very important
# * 3) A 1 unit increase in Avg. Session Length increases 26.03 units in Yearly Amount Spent --> important
# * 4) A 1 unit increase in Time on Website increases 0.36 units in Yearly Amount Spent --> least important

# ### The data indicates that if the company had to choose between focusing in its app or in its website, the app could bring greater returns.

# ### However, the data also indicates the most important feature is Length of Membership, meaning the company should focus their efforts primarily in customer retention strategies, such as:
# * Loyalty program
# * Personalized communications
# * Personalized offers
# * Re-engage customers through marketing automation
# * Provide Excellent Customer Service (direct contact, feedback, surveys, social media monitoring) 