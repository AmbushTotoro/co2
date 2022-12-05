#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np
import joblib
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[98]:


df = pd.read_csv(r"C:\Users\dc872\Desktop\Database2.csv")


# In[99]:


df.head(10)


# In[100]:


x = df.drop(['co2'],axis=1).values
y = df['co2'].values


# In[106]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.8, random_state=0)


# In[107]:


ml=LinearRegression()
ml.fit(x_train,y_train)


# In[108]:


y_pred=ml.predict(x_test)
print(y_pred)


# In[109]:


ml.predict([[300,50,0,50,73857768.0]])


# In[110]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[112]:


pyplot.figure(figsize=(15,10))
pyplot.scatter(y_test,y_pred)
pyplot.xlabel('Actual')
pyplot.ylabel('Predicted')
pyplot.title('Actual vs. Predicted')


# In[115]:


pred_y_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred, 'Difference': y_test-y_pred})
pred_y_df[0:20]


# In[ ]:




