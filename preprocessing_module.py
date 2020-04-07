#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score as r2
from collections import Counter
from scipy import stats
from math import sqrt
from math import log
import pickle
import category_encoders as ce
from sklearn.preprocessing import StandardScaler


# In[2]:


df=pd.read_csv('listings1.csv')
df=df.iloc[np.random.permutation(len(df))]
df=df.drop(['host_response_rate','security_deposit','cleaning_fee'],axis=1)
df=df.drop_duplicates()
df=df.reset_index()
df=df.drop('index',axis=1)


# In[3]:


pt.figure(figsize=(7,6))
pt.title('Room types that are rented')
sb.countplot(x=df['room_type'])


# In[4]:


train_df = df.iloc[0:9071,:]
test_df = df.iloc[9071:,:]
test_df=test_df.reset_index()
test_df=test_df.drop('index',axis=1)


# In[5]:


print('Data splitted into Training and Test:')
print('Training Dataframe shape: ',train_df.shape)
print('Testing Dataframe shape: ',test_df.shape)


# In[6]:


pt.bar(df['accommodates'],df['price'])
pt.xlabel('Number of guests that can be accomodated')
pt.ylabel('Price of the listing')
pt.title('Guest Capacity Vs Price of the listing')
pt.show()


# In[7]:


pt.barh(df['guests_included'],df['price'])
pt.xlabel('Number of guests allowed')
pt.ylabel('Price of the listing')
pt.title('Guest allowed Vs Price of the listing')
pt.show()


# In[8]:


pt.title('Number of reviews Vs Price of the listing')
sb.scatterplot(x=df['number_of_reviews'],y=df['price'])


# In[9]:


pt.title('Number of reviews Vs Price of the listing')
sb.scatterplot(x=df['bedrooms'],y=train_df['price'],hue=df['room_type'])


# dropping some columns because of the missing values
# dividing into training and test set

# #Onehot encoding of the categorical variables

# In[10]:


encoder = ce.OneHotEncoder(cols=['room_type','bed_type'], handle_unknown='indicator')
train_df = encoder.fit_transform(train_df)
test_df = encoder.fit_transform(test_df)


# In[11]:


train_df=train_df.drop(['bed_type_5', 'bed_type_-1'],axis=1)
test_df=test_df.drop(['bed_type_5', 'bed_type_-1'],axis=1)
print('Dataframes after one hot encoding of categorical features:')
print('Training Dataframe shape: ',train_df.shape)
print('Testing Dataframe shape: ',test_df.shape)


# Filling the missing values of the bedrooms and bathrooms with the mode of the respective features

# In[12]:


train_df['bedrooms']=train_df['bedrooms'].fillna(1)
train_df['bathrooms']=train_df['bathrooms'].fillna(1)
test_df['bedrooms']=test_df['bedrooms'].fillna(1)
test_df['bathrooms']=test_df['bathrooms'].fillna(1)


# removing the points where the price is equal to zero

# In[13]:


train_df=train_df.drop(train_df[train_df['price']==0].index,axis=0)
test_df=test_df.drop(test_df[test_df['price']==0].index,axis=0)
train_df=train_df.reset_index()
train_df=train_df.drop('index',axis=1)
test_df=test_df.reset_index()
test_df=test_df.drop('index',axis=1)


# In[14]:


pt.figure(figsize=(10,5))
sb.distplot(train_df['price'],bins=30)


# In[15]:


pt.figure(figsize=(10,5))
sb.distplot(test_df['price'],bins=30)


# In[16]:


z_score = np.abs(stats.zscore(train_df['price']))
outlier_index=np.where(z_score>3)
for item in outlier_index[0]:
    train_df=train_df.drop(item,axis=0)


# In[17]:


z_score = np.abs(stats.zscore(test_df['price']))
outlier_index2=np.where(z_score>3)
for item in outlier_index2[0]:
    test_df=test_df.drop(item,axis=0)


# In[18]:


print('Data after removing the zero values and outliers in price in Training and Test Dataset:')
print('Training Dataframe shape: ',train_df.shape)
print('Testing Dataframe shape: ',test_df.shape)


# In[19]:


train_df=train_df.sort_values(by=['host_total_listings_count'])
train_df=train_df.reset_index()
train_df=train_df.drop('index',axis=1)


# In[20]:


value=train_df.shape[0]-train_df['host_total_listings_count'].isnull().sum(axis=0)


# In[21]:


knn_train_df=train_df[0:value]
#actual_train_df=knn_train_df
knn_test2_df=train_df[value:]


# In[22]:


print(knn_train_df.shape)
print(knn_test2_df.shape)


# In[23]:


z_score = np.abs(stats.zscore(knn_train_df['host_total_listings_count']))
outlier_index=np.where(z_score>3)
for item in outlier_index[0]:
    knn_train_df=knn_train_df.drop(item,axis=0)
actual_train_df=knn_train_df


# In[24]:


knn_train_df=knn_train_df.reset_index()
y_train=knn_train_df['host_total_listings_count']
knn_train_df=knn_train_df.drop(['host_total_listings_count','index'],axis=1)
knn_test_df=knn_test2_df.drop(['host_total_listings_count'],axis=1)


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(knn_train_df, y_train, test_size=0.3, random_state=42)


# In[26]:


training_err=[]
test_err=[]
from sklearn.neighbors import KNeighborsClassifier
for i in range(1,31):
    classifier=KNeighborsClassifier(n_neighbors=i,metric='manhattan')
    classifier.fit(X_train,y_train)
    pred_i=classifier.predict(X_train)
    training_error=np.mean(pred_i != y_train)
    pred=classifier.predict(X_test)
    test_error=np.mean(pred != y_test)
    training_err.append(training_error)
    test_err.append(test_error)
    #print(i,' ',training_error,' ',test_error)


# In[27]:


best_k=np.argmin(test_err)+1
classifier=KNeighborsClassifier(n_neighbors=best_k,metric='manhattan')
classifier.fit(X_train,y_train)
pred=classifier.predict(knn_test_df)


# In[28]:


knn_test2_df=knn_test2_df.reset_index()
knn_test2_df=knn_test2_df.drop(['index'],axis=1)
knn_test2_df[0:]['host_total_listings_count']=list(pred)


# In[29]:


Final_df=pd.concat((actual_train_df,knn_test2_df),axis=0)
Final_df=Final_df.reset_index()
Final_df=Final_df.drop(['index'],axis=1)


# In[30]:


test_df['host_total_listings_count']=test_df['host_total_listings_count'].fillna(1)


# In[31]:


corr = Final_df.corr()
corr_features=set()
for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > 0.8:
            columnname = corr.columns[i]
            corr_features.add(columnname)


# In[32]:


preprocessed_train_df=Final_df.drop(['beds'],axis=1)
preprocessed_test_df=test_df.drop(['beds'],axis=1)
preprocessed_train_df=preprocessed_train_df.iloc[np.random.permutation(len(preprocessed_train_df))]
training_set=preprocessed_train_df[0:7023]
validation_set=preprocessed_train_df[7023:]
training_set.to_csv(path_or_buf='/Users/mohan/Downloads/EE660-Project/training set.csv')
validation_set.to_csv(path_or_buf='/Users/mohan/Downloads/EE660-Project/validation set.csv')
preprocessed_test_df.to_csv(path_or_buf='/Users/mohan/Downloads/EE660-Project/testing set.csv')


# In[ ]:




