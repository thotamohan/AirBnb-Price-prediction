## AirBnb-Price-prediction

# **1.Abstract**

The price prediction is crucial to the owners of the website, the property listers and the customers of the website to make informed decisions and in the modifications of the real estate policies. The price of the property listed on Airbnb is dependent on various factors such as Bedrooms, room types, number of guests and many others. The proposed study analyzes the price prediction capabilities of various Machine Learning Algorithms such as Linear Regression, Random Forest Regression, Support Vector Regression, Ridge Regression, and Adaboost Regression. Hyperparameter tuning is done on the models which performed better on the Validation Set. The models are evaluated using four metrics namely mean squared error, mean absolute error, r2 scores, and root mean squared error.         
**Keywords**: Regression, Random Forest, Regularization, Airbnb Dataset, Listing, Prediction, Boosting

# **2. Introduction**
# **2.1 Problem Type, Statement and Goals** 
The study proposed aims to predict the nightly price of the one stay in the Airbnb listings in the city of Austin. The price of an Airbnb listing is based on a variety of features namely Room type, number of beds, bathrooms, number of guests that can be accommodated, etc. Data is analyzed by various approaches to evaluating the importance of the features in the accurate price prediction of the listing. Various Machine Learning algorithms like Linear Regression, Ridge Regression, Support Vector Regression, Random Forest Regression, and Support Vector Regression are utilized for price prediction. Good performing models are selected from the above algorithms by the performance on the validation set.  Hyperparameter tuning is done on the selected models to obtain the best parameters for the respective models. 

Aim of price prediction is trivial, as all the properties listed are not owned by Airbnb. It acts as a medium to fill the gap between the vacant properties available and the customers looking for places to stay. Apart from that it also acts as a medium for the 
property listers to showcase their properties to a much bigger audience and get them filled. The proposed study act as a Recommender system for the property listers to fix price for their listings so that the fixed rate is feasible and that on par with other competitors. 

The complexity of the approach arises as the data available has many missing values, there are a lot of outliers in the price and other features, the high dimensionality of the features, there are also categorical features in the data. Therefore, data needs to be preprocessed, categorical features need to be encoded and the missing values to be filled with respect to the data filling techniques before the application of the above-mentioned algorithms. The problem of high dimensionality can be handled by introducing the random forest regression which analyzes the importance of features by a metric. From the above, we could see that the main challenges in the accurate price prediction are preprocessing of the data, dimensionality reduction, analyzing the performance of the various models and hyperparameter tuning of the selected models. 

# **2.2 Overview of Approach**
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

# **3.1 Data Set**
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
