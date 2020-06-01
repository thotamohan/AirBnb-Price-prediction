# Price-prediction of Airbnb properties 

## **1.Abstract**

The price prediction is crucial to the owners of the website, the property listers and the customers of the website to make informed decisions and in the modifications of the real estate policies. The price of the property listed on Airbnb is dependent on various factors such as Bedrooms, room types, number of guests and many others. The proposed study analyzes the price prediction capabilities of various Machine Learning Algorithms such as Linear Regression, Random Forest Regression, Support Vector Regression, Ridge Regression, and Adaboost Regression. Hyperparameter tuning is done on the models which performed better on the Validation Set. The models are evaluated using four metrics namely mean squared error, mean absolute error, r2 scores, and root mean squared error.         
**Keywords**: Regression, Random Forest, Regularization, Airbnb Dataset, Listing, Prediction, Boosting

# **2. Introduction**
## **2.1 Problem Type, Statement and Goals** 
The study proposed aims to predict the nightly price of the one stay in the Airbnb listings in the city of Austin. The price of an Airbnb listing is based on a variety of features namely Room type, number of beds, bathrooms, number of guests that can be accommodated, etc. Data is analyzed by various approaches to evaluating the importance of the features in the accurate price prediction of the listing. Various Machine Learning algorithms like Linear Regression, Ridge Regression, Support Vector Regression, Random Forest Regression, and Support Vector Regression are utilized for price prediction. Good performing models are selected from the above algorithms by the performance on the validation set.  Hyperparameter tuning is done on the selected models to obtain the best parameters for the respective models. 

Aim of price prediction is trivial, as all the properties listed are not owned by Airbnb. It acts as a medium to fill the gap between the vacant properties available and the customers looking for places to stay. Apart from that it also acts as a medium for the 
property listers to showcase their properties to a much bigger audience and get them filled. The proposed study act as a Recommender system for the property listers to fix price for their listings so that the fixed rate is feasible and that on par with other competitors. 

The complexity of the approach arises as the data available has many missing values, there are a lot of outliers in the price and other features, the high dimensionality of the features, there are also categorical features in the data. Therefore, data needs to be preprocessed, categorical features need to be encoded and the missing values to be filled with respect to the data filling techniques before the application of the above-mentioned algorithms. The problem of high dimensionality can be handled by introducing the random forest regression which analyzes the importance of features by a metric. From the above, we could see that the main challenges in the accurate price prediction are preprocessing of the data, dimensionality reduction, analyzing the performance of the various models and hyperparameter tuning of the selected models. 

## **2.2 Overview of Approach**
The approach presented here is to utilize the given data efficiently and predict the
price of the listing. Here features are not removed without any idea, they are
analyzed before removing. Like say, number of beds plays an important role in price
prediction of the listing so it wont be disregarded. But in order to deal with the
categorical features, one hot encoding is used but the features are then analyzed by
the random forest and identified the features which are helpful in the regression.
Firstly, the dataset is divided into the training and test sets, and then training set is
preprocessed and all the missing data is filled, outliers are removed, categorical
features are encoded, and the same for test data too. Then the processed training set
is then divided into training and validation. 
Then various machine Learning models are used, they are trained on the training set
and then performance is evaluated on validation set. Metrics like MSE, R2 Score,
MAE are used for evaluation. Then the best performing models on the validation set
are used for hyper parameter tuning and the model performance is calculated on the
actual test set.

# **3.Dataset Description**
## **3.1 Data Set**
Dataset consists of 11,340 points and 16 features. Number of features like count of
Host total listings, Room type, Number of bedrooms, beds and most of the
important information about the listing is available. Out of all the feature two are
categorical and the other are non-categorical. The table below gives us a better
understanding of the features of the data and what they mean
|Features | Description | Data type |
|---------|-------------|-----------|
| Host listings count | Count of the listings owned by host | float64|
| Room type | Type of the property | object |
| Host response rate | Response of the host for any service ranked | Int64 |
| Security deposit | Deposit charged by the host | float64 |
|accommodates |Number of guests can be accommodated |Int64|
|bathrooms |Number of bathrooms|float64|
|bedrooms |Number of bedrooms |float64|
|beds| Number of beds in the room| Int64
|Bed type| Type of the bed in room| object|
|price |Cost of the listing per night |Int64|
|Guests included |Number of guests for the stay |Int64|
|Minimum nights| Minimum nights that can be stayed |Int64|
|Maximum nights| Maximum nights that can be stayed| Int64|
|Availability 365 |Number of days available in an year |Int64|
|Number of reviews |Number of reviews given per listing |Int64|
|Cleaning fee |Fee collected for cleaning per stay |float64|

## **3.2 Implementation**
Dataset used is the Airbnb open source listing data, Austin. Here the aim is to
predict the price of the listing per night in the city of Austin. The steps below
explains about the steps involved in achieving our aim of prediction.
* Dataset divided into Training and Test Set.
* Preprocessing done on the training and test set.
* Training set is divided into Training and Validation Set.
* Models are trained on the training set and evaluated on the validation set.
* Hyper parameter tuning on the best models and feature selection.
* Tuned model applied on the Test Dataset.

## **3.3 Preprocessing, Feature Extraction, Dimensionality Adjustment**
Out of the entire Dataset of 11,340 points, 20% of the dataset that is 2268 points are
used for the testing. Now the preprocessing is done on both the training and test set
separately. Firstly, dataset is randomly shuffled, even before the separation of the
training and test set. Preprocessing steps involved are
1. All the duplicates in the data frame are dropped. Now the training and test sets
are separated.
2. Now only 18 datapoints bathroom value have missing values, therefore they are
replaced with the mode in both training and test set.
3. Now only 7 datapoints bedroom value have missing values, therefore they are
replaced with the mode in both training and test set.
4. As we could see that, there are two categorical features, which are encoded to
use in Machine Learning, but some of the classes in the categorical features are
very less in number when compared to the entire dataset therefore they are
removed. Like one of the class in the categorical feature ‘bed type’ has only 19
occurrences when compared to the entire dataset of 11340 points.
5. Now the datapoints which have a price value equal to zero are removed.
6. Then the outliers in the price value are removed for better learning for both
training and test set.
7. In the same way, outliers in host total listings count are removed for better
prediction using knn.
8. But the feature host total listing count has 348 missing values, for which knn is
used to fill the missing values.
a) For Training set:
Entire training set is divided into two points, one is for training knn which
does not have any missing value in host total listing count. And the other
is used to predict host total listing count values using this model.
b) For Test set:
The host total listing count is replaced with the mode of the entire dataset.
It is not used with knn.
9. With the resulted features, Correlation analysis is done, then the features are
removed with a value greater than 0.8.
10. Now the training set is divided into training and validation set i.e., the
processed training set is divided into 20%(1761 points) as validation set and the
rest as training set which are then imported to a csv file to use them in the main
file for models.

## **3.4 Dataset Methodology**
1. Firstly, dataset is randomly shuffled, even before the separation of the training
and test set. Out of the entire Dataset of 11,340 points, 20% of the dataset that
is 2268 points are used for the testing. Preprocessing is done on the training and
test separately. the processed training set is divided into 20% (1761 points) as
validation set and the rest as training set which are then imported to a csv files
separately to use them in the main file for models.
2. Now we have a training, test and validation set where training set is used to
train the machine learning model and it is used to test on validation set. A total
of six models are used in the price prediction. The model performances are
calculated on the validation set and the results are evaluated using metrics
namely MSE, RMSE, R2 score.
3. Out of the six, best performing two models on validation set are selected, Now
the two models are trained using the entire training set (both the training and
validation set) and are tested on the test set. Metrics used are same as above.

# **3.5 Training Process**

## 1. Linear Regression
Linear Regression tries to minimize the sum of square of the error between the
observed values and predicted values by linear approximation.
For the implementation of the model, scikit-learn is used, firstly default
parameters provided by scikit-learn are used to train on the training set and then
evaluated on validation set. Singular value decomposition of X is used to
compute the solution by Linear Regression.

## 2. Ridge Regression
Ridge Regression is used, because if there is non linear relationship between the
independent variables which makes the least square estimates unbiased and large
variances resulting in huge gap between the observed and predicted values. Ridge
regression handles this non linearity by adding a degree of bias to the estimator
which reduces the standard errors.

## 3. Lasso Regression
Lasso Regression is also similar like Linear Regression, but it uses shrinkage in
order to reduce the multicollinearity between independent variables. This selects
the features which are best for the regression shrinking the unimportant feature
weight values to zero.

## 4. Random Forest regression
Several decision trees are created by randomness and they are averaged to obtain
the final regressor. A random set of samples are drawn with replacement while
training the final regressor.
This model also performs feature selection by calculating the feature importance
and then leaving the features with value equal to zero.

## 5. Adaboost Regression
Adaboost fits a sequence of weak learners on data which is modified repeated.
Weak learners here refer to the small decision trees which predict better than
random guessing. Then weighted majority vote is done to combine all the
prediction, so that a final prediction is made. This property might help in better
prediction results.

# **3.6 Selection and Comparison of Results**

Out of the models mentioned above, we could see that the Random Forest Regression and Adaboost regression seemed to have performed well with the default parameters. Hyperparameter tuning is done on the these models to even obtain the better performance on the training and test set.
## • Random Forest Model
Randomized searchCV from scikit learn is used to obtain the best hyper parameters for the random forest model, the obtained parameters from the abovementioned functions are 200 number of estimators, maximum depth is 5 and criterion used is ‘mse’.
## • Lasso Model
Randomized searchCV from scikit learn is used to obtain the best hyper parameters for the Lasso model, the obtained parameters from the abovementioned functions are 0.539 alpha value, fit intercept is true. 
These tuned models are then trained on the entire training set which includes both training and validation set and tested on actual test data. Then features are selected using Random Forest method and then the models are trained on selected features and tested on actual test data.

## 3.6.1 **Final Results and Interpretation**

|Model|Training data|Validation data|
|-----|-------------|---------------|
|**Linear Regression**|MSE: 72694.07822283143|MSE: 75974.2332240539|
|   |MAE: 146.9261989455283|MAE: 149.79967745014662|
|   |RMSE: 269.61839370271355|RMSE: 275.63423812010996|
|   | R2: -0.6597861522972825|R2: -0.7017597561047282|
|**Ridge Regression**|MSE: 72674.17338993505|MSE: 75983.84420800908|
|   |MAE: 146.59165683796658|MAE: 149.66798563543358|
|   |RMSE: 269.58147820266703|RMSE: 275.65167187595483|
|   | R2: -0.6528843390107788|R2: -0.6924966300937911|
|**Support Vector Regression**|MSE: 102386.52705267385|MSE: 113741.25303304457|
|   |MAE: 138.00581509912323|MAE: 143.31278896261537|
|   |RMSE: 319.9789478273123|RMSE: 337.2554714649483|
|   | R2: -15.617948354346481|R2: -17.18405102245768|
|**Random Forest Regression**|MSE: 10450.919193931823|MSE: 57163.49713989214|
|   |MAE: 50.54620852374578|MAE: MAE: 124.25965257849114|
|   |RMSE:102.22973732692373|RMSE: 239.0888896203505|
|   | R2: 0.8791710668149424|R2: 0.30185910807466376|
|**Lasso Regression**|MSE: 72687.80452118929|MSE: 76011.81790707962|
|   |MAE: 146.2563537956556|MAE: 149.25157396442634|
|   |RMSE: 269.58147820266703|RMSE: 275.70240823590865|
|   | R2: -0.6767918627724787|R2: -0.7181451111942903|
|**Adaboost Regression**|MSE: 94545.81361038845|MSE: 96140.69133130621|
|   |MAE: 190.20023528698502|MAE: 194.56000295985336|
|   |RMSE: 307.48302979252117|RMSE: 310.06562423349385|
|   | R2: 0.19921550054727677|R2: 0.2197584720208533|


## 3.6.2 **Tuned Models**:

|Tuned Model|Training dataset|Test dataset|
|-----------|----------|----------|
|**Random Forest Regression**|MSE: 50119.60200790658|MSE: 72162.94845239061|
|   |MAE: 121.03146505838296|MAE: 72162.94845239061|
|   |RMSE: 223.87407623015795|RMSE: 268.6316222122604|
|   | R2: 0.11414507716244404|R2: -0.3659939487905348|
|**Adaboost Regression**|MSE: 62789.487541620794|MSE: 79804.39693832299|
|   |MAE: 139.7155368840744|MAE: 147.52761854440104|
|   |RMSE: 250.57830620710325|RMSE: 282.4967202257806|
|   | R2: -0.29382758329712644|R2: -0.7112303651306913|

## 3.6.3 **Performance of models after feature selection by Random Forest**:
|Tuned Model|Training dataset|Test dataset|
|-----------|----------|----------|
|**Random Forest Regression**|MSE: 50380.92308286226|MSE: 72699.10403067133|
|   |MAE: 121.628914377444|MAE: 134.99902596248782|
|   |RMSE: 224.4569515137864|RMSE: 269.62771376598386|
|   | R2: 0.10752789914307137|R2: -0.3523680334116106|
|**Adaboost Regression**|MSE: 62990.45746237391|MSE: 80049.17783907466|
|   |MAE: 139.95245299387756|MAE: 147.9597960228515|
|   |RMSE: 250.9789980503825|RMSE:282.9296340772289|
|   | R2: -0.3254053327707984|R2: -0.7484804416776716|

## 3.6.4 **Performance of models after feature selection by Lasso technique**:
|Tuned Model|Training dataset|Test dataset|
|-----------|----------|----------|
|**Random Forest Regression**|MSE: 50074.08993247161|MSE: 73209.09467790202|
|   |MAE: 120.98583893348983|MAE: 134.7862854694249|
|   |RMSE: 223.77240654842055|RMSE: 270.5717920957431|
|   | R2: 0.11262466988192954|R2: -0.3989555010925063|
|**Adaboost Regression**|MSE: 62463.85391790112|MSE: 79572.82290430536|
|   |MAE: 139.93383096384247|MAE: 148.01160643025068|
|   |RMSE: 249.92769738046465|RMSE:282.0865521507634|
|   | R2: -0.26101104979870016|R2: R2: -0.6754950063472749|

# 4 **Summary** 
## 4.1 **Conclusions and Interpretations**
1. The results of the feature selection showed that not all features are necessary. Even with less number of features we can obtain better accuracy.
2. 	My results show that the Support vector regression, Ridge Regression, Linear Reg. failed to perform well. Because of the non-linearity in the data.
3. 	Preprocessing plays a better role in obtaining best results.
4. 	Validation set performance can be generalized to the actual test.
5. 	Usage of models on validation set helps us in finding better models rather than trying on entire dataset.
6. After hyper parameter tuning of the random forest and Adaboost reg. there is a slight increase in the performance but there is a increase in the training error because of generalization.
7. Lasso and random forest feature selection yield better results even with lesser features and reduce computational power.
8. Hyper parameter tuning plays a better role in getting optimal result that can perform better on test data.

# **References:**

Special thanks to Prof. Keith Jenkins at University of Southern California for helping me in getting a clear understanding of the BFR algorithm both theoritically and practically.

