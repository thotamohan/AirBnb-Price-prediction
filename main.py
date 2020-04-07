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
from sklearn.linear_model import Lasso
from sklearn import linear_model
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
import warnings
warnings.filterwarnings("ignore")

# In[2]:


preprocessed_train_df1=pd.read_csv('training set.csv')
preprocessed_train_df=preprocessed_train_df1.drop(['Unnamed: 0'],axis=1)
train_price=preprocessed_train_df['price']
preprocessed_train_df=preprocessed_train_df.drop(['price'],axis=1)
preprocessed_test_df=pd.read_csv('testing set.csv')
preprocessed_test_df=preprocessed_test_df.drop(['Unnamed: 0'],axis=1)
test_price=preprocessed_test_df['price']
preprocessed_test_df=preprocessed_test_df.drop(['price'],axis=1)
validation_df1=pd.read_csv('validation set.csv')
validation_df=validation_df1.drop(['Unnamed: 0'],axis=1)
validation_price=validation_df['price']
validation_df=validation_df.drop(['price'],axis=1)


# In[3]:


print('Shapes of the training, validation and test dataframes')
print('Training dataframe: ',preprocessed_train_df.shape)
print('validation dataframe: ',validation_df.shape)
print('Testing dataframe: ',preprocessed_test_df.shape)


# In[4]:


scalar=StandardScaler()
X_norm=scalar.fit(preprocessed_train_df)
X_train_norm=scalar.transform(preprocessed_train_df)
X_validation_norm=scalar.transform(validation_df)
X_test_norm=scalar.transform(preprocessed_test_df)


# In[5]:


print('Performance of the models on the training and validation data without hyper parameter tuning:')


# In[6]:


Lin_Reg=LinearRegression()
Lin_Reg.fit(X_train_norm,train_price)
y_train_pred=Lin_Reg.predict(X_train_norm)
y_validation_pred=Lin_Reg.predict(X_validation_norm)
LR_MSE_train=mse(y_train_pred,train_price)
LR_MSE_validation=mse(y_validation_pred,validation_price)
LR_R2_train=r2(y_train_pred,train_price)
LR_R2_validation=r2(y_validation_pred,validation_price)
LR_MAE_train=mae(y_train_pred,train_price)
LR_MAE_validation=mae(y_validation_pred,validation_price)
print('Linear Regression model MSE on train data :' ,LR_MSE_train)
print('Linear Regression model MSE on validation data :' ,LR_MSE_validation)
print('Linear Regression model R2 score on train data :' ,LR_R2_train)
print('Linear Regression model R2 score on validation data :' ,LR_R2_validation)
print('Linear Regression model RMSE on train data :' ,sqrt(LR_MSE_train))
print('Linear Regression model RMSE on validation data :' ,sqrt(LR_MSE_validation))
print('Linear Regression model MAE on train data :' ,LR_MAE_train)
print('Linear Regression model MAE on validation data :' ,LR_MAE_validation)


# In[7]:


Ridge_Reg=Ridge()
Ridge_Reg.fit(X_train_norm,train_price)
y_train_pred=Ridge_Reg.predict(X_train_norm)
y_validation_pred=Ridge_Reg.predict(X_validation_norm)
Ridge_MSE_train=mse(y_train_pred,train_price)
Ridge_MSE_validation=mse(y_validation_pred,validation_price)
Ridge_R2_train=r2(y_train_pred,train_price)
Ridge_R2_validation=r2(y_validation_pred,validation_price)
Ridge_MAE_train=mae(y_train_pred,train_price)
Ridge_MAE_validation=mae(y_validation_pred,validation_price)
print('Ridge Regression model MSE on train data :' ,Ridge_MSE_train)
print('Ridge Regression model MSE on validation data :' ,Ridge_MSE_validation)
print('Ridge Regression model R2 score on train data :' ,Ridge_R2_train)
print('Ridge Regression model R2 score on validation data :' ,Ridge_R2_validation)
print('Ridge Regression model RMSE on train data :' ,sqrt(Ridge_MSE_train))
print('Ridge Regression model RMSE on validation data :' ,sqrt(Ridge_MSE_validation))
print('Ridge Regression model MAE on train data :' ,Ridge_MAE_train)
print('Ridge Regression model MAE on validation data :' ,Ridge_MAE_validation)


# In[8]:


rf=RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(X_train_norm,train_price)
y_train_pred=rf.predict(X_train_norm)
y_validation_pred=rf.predict(X_validation_norm)
RF_MSE_train=mse(y_train_pred,train_price)
RF_MSE_validation=mse(y_validation_pred,validation_price)
RF_R2_train=r2(y_train_pred,train_price)
RF_R2_validation=r2(y_validation_pred,validation_price)
RF_MAE_train=mae(y_train_pred,train_price)
RF_MAE_validation=mae(y_validation_pred,validation_price)
print('RandomForest Regression model MSE on train data :' ,RF_MSE_train)
print('RandomForest Regression model MSE on validation data :' ,RF_MSE_validation)
print('RandomForest Regression model R2 score on train data :' ,RF_R2_train)
print('RandomForest Regression model R2 score on validation data :' ,RF_R2_validation)
print('RandomForest Regression model RMSE on train data :' ,sqrt(RF_MSE_train))
print('RandomForest Regression model RMSE on validation data :' ,sqrt(RF_MSE_validation))
print('RandomForest Regression model MAE on train data :' ,RF_MAE_train)
print('RandomForest Regression model MAE on validation data :' ,RF_MAE_validation)


# In[9]:


pre_Lasso=linear_model.Lasso()
pre_Lasso.fit(X_train_norm,train_price)
y_train_pred=pre_Lasso.predict(X_train_norm)
y_validation_pred=pre_Lasso.predict(X_validation_norm)
Lasso_MSE_train=mse(y_train_pred,train_price)
Lasso_MSE_validation=mse(y_validation_pred,validation_price)
Lasso_R2_train=r2(y_train_pred,train_price)
Lasso_R2_validation=r2(y_validation_pred,validation_price)
Lasso_MAE_train=mae(y_train_pred,train_price)
Lasso_MAE_validation=mae(y_validation_pred,validation_price)
print('Lasso Regression model MSE on train data :' ,Lasso_MSE_train)
print('Lasso Regression model MSE on validation data :' ,Lasso_MSE_validation)
print('Lasso Regression model R2 score on train data :' ,Lasso_R2_train)
print('Lasso Regression model R2 score on validation data :' ,Lasso_R2_validation)
print('Lasso Regression model RMSE on train data :' ,sqrt(Ridge_MSE_train))
print('Lasso Regression model RMSE on validation data :' ,sqrt(Lasso_MSE_validation))
print('Lasso Regression model MAE on train data :' ,Lasso_MAE_train)
print('Lasso Regression model MAE on validation data :' ,Lasso_MAE_validation)


# In[10]:


S_vector=SVR(kernel='rbf')
S_vector.fit(X_train_norm,train_price)
y_train_pred=S_vector.predict(X_train_norm)
y_validation_pred=S_vector.predict(X_validation_norm)
SVR_MSE_train=mse(y_train_pred,train_price)
SVR_MSE_validation=mse(y_validation_pred,validation_price)
SVR_R2_train=r2(y_train_pred,train_price)
SVR_R2_validation=r2(y_validation_pred,validation_price)
SVR_MAE_train=mae(y_train_pred,train_price)
SVR_MAE_validation=mae(y_validation_pred,validation_price)
print('SupportVector Regression model MSE on train data :' ,SVR_MSE_train)
print('SupportVector Regression model MSE on validation data :' ,SVR_MSE_validation)
print('SupportVector Regression model R2 score on train data :' ,SVR_R2_train)
print('SupportVector Regression model R2 score on validation data :' ,SVR_R2_validation)
print('SupportVector Regression model RMSE on train data :' ,sqrt(SVR_MSE_train))
print('SupportVector Regression model RMSE on validation data :' ,sqrt(SVR_MSE_validation))
print('SupportVector Regression model MAE on train data :' ,SVR_MAE_train)
print('SupportVector Regression model MAE on validation data :' ,SVR_MAE_validation)


# In[11]:


Adaboost=AdaBoostRegressor()
Adaboost.fit(X_train_norm,train_price)
y_train_pred=Adaboost.predict(X_train_norm)
y_validation_pred=Adaboost.predict(X_validation_norm)
AB_MSE_train=mse(y_train_pred,train_price)
AB_MSE_validation=mse(y_validation_pred,validation_price)
AB_R2_train=r2(y_train_pred,train_price)
AB_R2_validation=r2(y_validation_pred,validation_price)
AB_MAE_train=mae(y_train_pred,train_price)
AB_MAE_validation=mae(y_validation_pred,validation_price)
print('AdaBoostRegression model MSE on train data :' ,AB_MSE_train)
print('AdaBoostRegression model MSE on validation data :' ,AB_MSE_validation)
print('AdaBoostRegression model R2 score on train data :' ,AB_R2_train)
print('AdaBoostRegression model R2 score on validation data :' ,AB_R2_validation)
print('AdaBoostRegression model RMSE on train data :' ,sqrt(AB_MSE_train))
print('AdaBoostRegression model RMSE on validation data :' ,sqrt(AB_MSE_validation))
print('AdaBoostRegression model MAE on train data :' ,AB_MAE_train)
print('AdaBoostRegression model MAE on validation data :' ,AB_MAE_validation)


# In[12]:


print('Hyperparameter tuning on Random Forest Model')


# In[13]:


entire_training_df=pd.concat([preprocessed_train_df1,validation_df1])
entire_training_df=entire_training_df.drop(['Unnamed: 0'],axis=1)
entire_training_price=entire_training_df['price']
entire_training_df=entire_training_df.drop(['price'],axis=1)
entire_X_norm=scalar.fit(entire_training_df)
entire_X_train_norm=scalar.transform(entire_training_df)


# In[14]:


#rf_tuned = pickle.load(open('rf_model.sav', 'rb'))
#rf_tuned.fit(entire_X_train_norm,entire_training_price)


# In[15]:


to_tune_rf=RandomForestRegressor()
depth=range(3,6)
n_estimators=[10, 50, 100, 200]
randomiser={'n_estimators': n_estimators,
            'max_depth': depth
            }
tuned_rf=RandomizedSearchCV(estimator = to_tune_rf, param_distributions = randomiser, n_iter = 200, cv = 3, verbose=False, random_state=42)
tuned_rf.fit(entire_X_train_norm,entire_training_price)
rf_tuned=tuned_rf.best_estimator_


# In[16]:


y_train_pred_tuned_rf=rf_tuned.predict(entire_X_train_norm)
y_test_pred_tuned_rf=rf_tuned.predict(X_test_norm)
tuned_rf_training_mse=mse(y_train_pred_tuned_rf,entire_training_price)
tuned_rf_testing_mse=mse(y_test_pred_tuned_rf,test_price)
tuned_rf_training_mae=mae(y_train_pred_tuned_rf,entire_training_price)
tuned_rf_testing_mae=mae(y_test_pred_tuned_rf,test_price)
tuned_rf_training_r2=r2(y_train_pred_tuned_rf,entire_training_price)
tuned_rf_testing_r2=r2(y_test_pred_tuned_rf,test_price)
print('Random Forest results after Hyper parameter tuning:')
print('Tuned RandomForest Regression model MSE on training data :' ,tuned_rf_training_mse)
print('Tuned RandomForest Regression model MSE on actual test data :' ,tuned_rf_testing_mse)
print('Tuned RandomForest Regression model R2 score on training data :' ,tuned_rf_training_r2)
print('Tuned RandomForest Regression model R2 score on actual test data :' ,tuned_rf_testing_r2)
print('Tuned RandomForest Regression model RMSE on training data :' ,sqrt(tuned_rf_training_mse))
print('Tuned RandomForest Regression model RMSE on actual test data :' ,sqrt(tuned_rf_testing_mse))
print('Tuned RandomForest Regression model MAE on training data :' ,tuned_rf_training_mae)
print('Tuned RandomForest Regression model MAE on actual test data :' ,tuned_rf_testing_mse)


# In[18]:


parameters = {'n_estimators': [50, 100],'learning_rate' : [0.01,0.05,0.1,0.3,1],'loss' : ['linear', 'square', 'exponential']}
Adaboost_tuning = RandomizedSearchCV(AdaBoostRegressor(),param_distributions = parameters,cv=3,n_iter = 10,n_jobs=-1)
Adaboost_tuning.fit(entire_X_train_norm,entire_training_price)
Adaboost_tuned=Adaboost_tuning.best_estimator_


# In[19]:


Adaboost_tuned=Adaboost_tuning.best_estimator_
y_train_pred_AB=Adaboost_tuned.predict(entire_X_train_norm)
y_test_pred_AB=Adaboost_tuned.predict(X_test_norm)
AB_MSE_train=mse(y_train_pred_AB,entire_training_price)
AB_MSE_test=mse(y_test_pred_AB,test_price)
AB_R2_train=r2(y_train_pred_AB,entire_training_price)
AB_R2_test=r2(y_test_pred_AB,test_price)
AB_MAE_train=mae(y_train_pred_AB,entire_training_price)
AB_MAE_test=mae(y_test_pred_AB,test_price)
print('Results on Adaboost Regression after Hyperparameter tuning')
print('AdaBoostRegression model MSE on train data :' ,AB_MSE_train)
print('AdaBoostRegression model MSE on test data :' ,AB_MSE_test)
print('AdaBoostRegression model R2 score on train data :' ,AB_R2_train)
print('AdaBoostRegression model R2 score on test data :' ,AB_R2_test)
print('AdaBoostRegression model RMSE on train data :' ,sqrt(AB_MSE_train))
print('AdaBoostRegression model RMSE on test data :' ,sqrt(AB_MSE_test))
print('AdaBoostRegression model MAE on train data :' ,AB_MAE_train)
print('AdaBoostRegression model MAE on test data :' ,AB_MAE_test)


# In[20]:


print('Feature selection by Random Forests')


# In[21]:


index_imp_random_features=np.where(rf.feature_importances_==0)
columns_train=entire_training_df.columns
columns_test=preprocessed_test_df.columns
rf_train_dataframe=entire_training_df
rf_test_dataframe=preprocessed_test_df
for item in index_imp_random_features[0]:
    rf_train_dataframe=rf_train_dataframe.drop(columns=columns_train[item])
    rf_test_dataframe=rf_test_dataframe.drop(columns=columns_test[item])
print('The shape of the dataframes after feature selection by Random Forest Technique')
print('Training dataframe:',rf_train_dataframe.shape)
print('Testing dataframe:',rf_test_dataframe.shape)


# In[22]:


print('Now applying the tuned models to the data after feature selection by random forests')


# In[23]:


preprocessed_norm=scalar.fit(rf_train_dataframe)
rf_train_dataframe_norm=scalar.transform(rf_train_dataframe)
rf_test_dataframe_norm=scalar.transform(rf_test_dataframe)


# In[24]:


#rf_tuned=RandomForestRegressor()
rf_tuned.fit(rf_train_dataframe_norm,entire_training_price)
y_train_pred_tuned=rf_tuned.predict(rf_train_dataframe_norm)
y_test_pred_tuned=rf_tuned.predict(rf_test_dataframe_norm)
tuned_rf_training_mse=mse(y_train_pred_tuned,entire_training_price)
tuned_rf_testing_mse=mse(y_test_pred_tuned,test_price)
tuned_rf_training_mae=mae(y_train_pred_tuned,entire_training_price)
tuned_rf_testing_mae=mae(y_test_pred_tuned,test_price)
tuned_rf_training_r2=r2(y_train_pred_tuned,entire_training_price)
tuned_rf_testing_r2=r2(y_test_pred_tuned,test_price)
print('The results on the data after feature selection by Random Forest Technique')
print('Tuned RandomForest Regression model MSE on entire train data :' ,tuned_rf_training_mse)
print('Tuned RandomForest Regression model MSE on actual test data :' ,tuned_rf_testing_mse)
print('Tuned RandomForest Regression model R2 score on entire train data :' ,tuned_rf_training_r2)
print('Tuned RandomForest Regression model R2 score on actual test data :' ,tuned_rf_testing_r2)
print('Tuned RandomForest Regression model RMSE on entire train data :' ,sqrt(tuned_rf_training_mse))
print('Tuned RandomForest Regression model RMSE on actual test data :' ,sqrt(tuned_rf_testing_mse))
print('Tuned RandomForest Regression model MAE on entire train data :' ,tuned_rf_training_mae)
print('Tuned RandomForest Regression model MAE on actual test data :' ,tuned_rf_testing_mae)


# In[26]:


Adaboost_tuned.fit(rf_train_dataframe_norm,entire_training_price)
y_train_pred=Adaboost_tuned.predict(rf_train_dataframe_norm)
y_test_pred=Adaboost_tuned.predict(rf_test_dataframe_norm)
AB_MSE_train=mse(y_train_pred,entire_training_price)
AB_MSE_test=mse(y_test_pred,test_price)
AB_R2_train=r2(y_train_pred,entire_training_price)
AB_R2_test=r2(y_test_pred,test_price)
AB_MAE_train=mae(y_train_pred,entire_training_price)
AB_MAE_test=mae(y_test_pred,test_price)
print('Results on Adaboost Regression after feature selection by Random Forests')
print('AdaBoostRegression model MSE on train data :' ,AB_MSE_train)
print('AdaBoostRegression model MSE on test data :' ,AB_MSE_test)
print('AdaBoostRegression model R2 score on train data :' ,AB_R2_train)
print('AdaBoostRegression model R2 score on test data :' ,AB_R2_test)
print('AdaBoostRegression model RMSE on train data :' ,sqrt(AB_MSE_train))
print('AdaBoostRegression model RMSE on test data :' ,sqrt(AB_MSE_test))
print('AdaBoostRegression model MAE on train data :' ,AB_MAE_train)
print('AdaBoostRegression model MAE on test data :' ,AB_MAE_test)


# In[27]:


Lasso_imp_features_index=np.where(pre_Lasso.coef_==0)
columns_train=entire_training_df.columns
columns_test=preprocessed_test_df.columns
lasso_train_dataframe=entire_training_df
lasso_test_dataframe=preprocessed_test_df
for item in Lasso_imp_features_index[0]:
    lasso_train_dataframe=lasso_train_dataframe.drop(columns=columns_train[item])
    lasso_test_dataframe=lasso_test_dataframe.drop(columns=columns_test[item])
print('The shape of the dataframes after feature selection by Lasso Technique')
print('Training dataframe:',lasso_train_dataframe.shape)
print('Testing dataframe:',lasso_test_dataframe.shape)


# In[28]:


preprocessed_norm=scalar.fit(lasso_train_dataframe)
lasso_train_dataframe_norm=scalar.transform(lasso_train_dataframe)
lasso_test_dataframe_norm=scalar.transform(lasso_test_dataframe)


# In[29]:


#rf_tuned=RandomForestRegressor()
rf_tuned.fit(lasso_train_dataframe,entire_training_price)
y_train_pred_tuned=rf_tuned.predict(lasso_train_dataframe)
y_test_pred_tuned=rf_tuned.predict(lasso_test_dataframe)
tuned_rf_training_mse=mse(y_train_pred_tuned,entire_training_price)
tuned_rf_testing_mse=mse(y_test_pred_tuned,test_price)
tuned_rf_training_mae=mae(y_train_pred_tuned,entire_training_price)
tuned_rf_testing_mae=mae(y_test_pred_tuned,test_price)
tuned_rf_training_r2=r2(y_train_pred_tuned,entire_training_price)
tuned_rf_testing_r2=r2(y_test_pred_tuned,test_price)
print('The results on the data after feature selection by Lasso Technique')
print('Tuned RandomForest Regression model MSE on entire train data :' ,tuned_rf_training_mse)
print('Tuned RandomForest Regression model MSE on actual test data :' ,tuned_rf_testing_mse)
print('Tuned RandomForest Regression model R2 score on entire train data :' ,tuned_rf_training_r2)
print('Tuned RandomForest Regression model R2 score on actual test data :' ,tuned_rf_testing_r2)
print('Tuned RandomForest Regression model RMSE on entire train data :' ,sqrt(tuned_rf_training_mse))
print('Tuned RandomForest Regression model RMSE on actual test data :' ,sqrt(tuned_rf_testing_mse))
print('Tuned RandomForest Regression model MAE on entire train data :' ,tuned_rf_training_mae)
print('Tuned RandomForest Regression model MAE on actual test data :' ,tuned_rf_testing_mae)


# In[30]:


#rf_tuned=RandomForestRegressor()
rf_tuned.fit(lasso_train_dataframe_norm,entire_training_price)
y_train_pred_tuned=rf_tuned.predict(lasso_train_dataframe_norm)
y_test_pred_tuned=rf_tuned.predict(lasso_test_dataframe_norm)
tuned_rf_training_mse=mse(y_train_pred_tuned,entire_training_price)
tuned_rf_testing_mse=mse(y_test_pred_tuned,test_price)
tuned_rf_training_mae=mae(y_train_pred_tuned,entire_training_price)
tuned_rf_testing_mae=mae(y_test_pred_tuned,test_price)
tuned_rf_training_r2=r2(y_train_pred_tuned,entire_training_price)
tuned_rf_testing_r2=r2(y_test_pred_tuned,test_price)
print('The results on the data after feature selection by Lasso Technique')
print('Tuned RandomForest Regression model MSE on entire train data :' ,tuned_rf_training_mse)
print('Tuned RandomForest Regression model MSE on actual test data :' ,tuned_rf_testing_mse)
print('Tuned RandomForest Regression model R2 score on entire train data :' ,tuned_rf_training_r2)
print('Tuned RandomForest Regression model R2 score on actual test data :' ,tuned_rf_testing_r2)
print('Tuned RandomForest Regression model RMSE on entire train data :' ,sqrt(tuned_rf_training_mse))
print('Tuned RandomForest Regression model RMSE on actual test data :' ,sqrt(tuned_rf_testing_mse))
print('Tuned RandomForest Regression model MAE on entire train data :' ,tuned_rf_training_mae)
print('Tuned RandomForest Regression model MAE on actual test data :' ,tuned_rf_testing_mae)


# In[32]:


Adaboost_tuned.fit(lasso_train_dataframe_norm,entire_training_price)
y_train_pred_Lasso=Adaboost_tuned.predict(lasso_train_dataframe_norm)
y_test_pred_Lasso=Adaboost_tuned.predict(lasso_test_dataframe_norm)
AB_MSE_train=mse(y_train_pred_Lasso,entire_training_price)
AB_MSE_test=mse(y_test_pred_Lasso,test_price)
AB_R2_train=r2(y_train_pred_Lasso,entire_training_price)
AB_R2_test=r2(y_test_pred_Lasso,test_price)
AB_MAE_train=mae(y_train_pred_Lasso,entire_training_price)
AB_MAE_test=mae(y_test_pred_Lasso,test_price)
print('Results on Adaboost Regression after feature selection by Lasso Technique')
print('AdaBoostRegression model MSE on train data :' ,AB_MSE_train)
print('AdaBoostRegression model MSE on test data :' ,AB_MSE_test)
print('AdaBoostRegression model R2 score on train data :' ,AB_R2_train)
print('AdaBoostRegression model R2 score on test data :' ,AB_R2_test)
print('AdaBoostRegression model RMSE on train data :' ,sqrt(AB_MSE_train))
print('AdaBoostRegression model RMSE on test data :' ,sqrt(AB_MSE_test))
print('AdaBoostRegression model MAE on train data :' ,AB_MAE_train)
print('AdaBoostRegression model MAE on test data :' ,AB_MAE_test)


# In[ ]:




