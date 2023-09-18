#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("Electric_Prod.csv",index_col=0,parse_dates=True)


# In[3]:


df


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


df['eprod'].plot(figsize=(20,5))
plt.show()


# In[6]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[7]:


#dec=seasonal_decompose(df['Electricity_production'],model="add")
#dec.plot();


# In[8]:


from pmdarima import auto_arima


# In[9]:


auto_arima(df['eprod'],seasonal=True,m=12).summary()


# In[10]:


df_train=df.iloc[:318]
df_test=df.iloc[318:]


# In[11]:


df_train.tail()


# In[12]:


df_test.head()


# In[13]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
model=SARIMAX(df['eprod'],order=(1,1,2),seasonal_order=(1,0,1,12))
re=model.fit()
re.summary()


# In[14]:


start=len(df_train)
end=len(df_train)+len(df_test)-1
predictions=re.predict(start=start,end=end,dynamic=False,typ="levels").rename("SARIMAX(1,1,2)x(1,0,1,12) Predictions")


# In[15]:


predictions


# In[16]:


for i in range(len(predictions)):
    print(f"predicted={predictions[i]:<11.20},expected={df_test['eprod'][i]}")


# In[17]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(df_test['eprod'],predictions)
print(f"SARIMAX(1,1,2)x(1,0,1,12) MSE Error:{mse:11.18}")


# In[18]:


import math


# In[19]:


RMSE=math.sqrt(mse)


# In[20]:


RMSE


# In[21]:


title="Electric Production"
ylabel=''
xlabel=''
ax=df_test['eprod'].plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis="x",tight=True)
ax.set(xlabel=xlabel,ylabel=ylabel)


# In[22]:


model=SARIMAX(df['eprod'],order=(1,1,2),seasonal_order=(1,0,1,12))
re=model.fit()
forecast=re.predict(len(df),len(df)+10,typ='levels').rename('SARIMAX(1,1,2)x(1,0,1,12) Forecast')


# In[23]:


forecast


# In[ ]:




