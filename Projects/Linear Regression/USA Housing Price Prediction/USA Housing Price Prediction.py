#!/usr/bin/env python
# coding: utf-8

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


# In[2]:


df = pd.read_csv('USA_Housing.csv')


# In[3]:


df.head()


# # Análise do Conteúdo das Variáveis

# In[4]:


# informações sobre conteúdo das variáveis
df.info()


# ## Checar Dados Faltantes

# In[5]:


sns.heatmap(data = df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# In[6]:


# Informações sobre variáveis numéricas
df.describe()


# In[7]:


# Útil para copiar o nome das colunas corretamente
df.columns


# # Análise Exploratória dos Dados --> EDA (Exploratory Data Analysis)

# In[8]:


# Para dataframes menores, pairplot é um bom overview dos dados
sns.pairplot(df)


# ## Variável Resposta --> Price

# In[9]:


sns.histplot(df['Price'], kde = True)


# In[10]:


# usar mapa de calor para observar correlações entre as variáveis
plt.figure(figsize=(15,5))
sns.heatmap(df.corr(), annot = True)


# # Divisão entre treino (x) e alvo (y)

# In[11]:


df.columns


# In[12]:


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[13]:


y = df['Price']


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[16]:


from sklearn.linear_model import LinearRegression


# In[17]:


lm = LinearRegression()


# In[18]:


lm.fit(X_train, y_train)


# In[19]:


print(lm.intercept_)


# In[20]:


lm.coef_


# In[21]:


X_train.columns


# In[22]:


cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])


# In[23]:


cdf


# # Predições

# In[24]:


predictions = lm.predict(X_test)


# In[25]:


# Predições de preço segundo o modelo
predictions


# In[26]:


# Dados reais de preço
y_test


# In[27]:


# Quanto mais alinhados os pontos, melhor o modelo
plt.scatter(y_test, predictions)


# In[28]:


# Representação gráfica dos resíduos --> IDENTIFICAR SE O MODELO ESCOLHIDO É ADEQUADO --> no caso, regressão linear
# Caso os resíduos se assemelhem a uma distribuição normal, é um evidência que o modelo escolhido é ideal para os dados
sns.histplot((y_test - predictions), kde = True)


# # Regression Evaluation Metrics --> Loss Functions

# ## 1) (MAE) --> Mean Absolute Error --> "mean of the squares errors"

# Basically the average error.

# ## 2) (MSE) --> Mean Squared Error --> mean of the squared errors"

# More poppular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.

# ## 3) (RMSE) --> Root Mean Squared Error --> "square root of the mean of the squared errors"

# Even more popular than MSE, because RMSE is interpretable in the "y" units --> target variable

# In[29]:


from sklearn import metrics


# In[30]:


mae = metrics.mean_absolute_error(y_test, predictions)


# In[31]:


mae


# In[32]:


mse = metrics.mean_squared_error(y_test, predictions)


# In[33]:


mse


# In[34]:


rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))


# In[35]:


rmse


# In[36]:


r2 = metrics.r2_score(y_test, predictions)


# In[37]:


r2


# In[38]:


indicators = pd.array(data=[mae,mse,rmse,r2])


# In[39]:


indicators


# In[40]:


pd.DataFrame(data = indicators, index = ['MAE','MSE','RMSE','R²'], columns = ['Values'])

