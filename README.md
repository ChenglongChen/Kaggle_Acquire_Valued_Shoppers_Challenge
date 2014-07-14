# Kaggle's Acquire Valued Shoppers Challenge
  
This repo holds the Python and R code I used to make submision to Kaggle's Acquire Valued Shoppers Challenge. The score using this implementation(s) is around 0.605~0.609.


## Method

* This competition is about dealing with "big data" (around 20G+) and mostly about feature engineering. To handling this amount of data and construct feature, I rely on Python. In specific, the corresponding code in the ./Python folder are based on Kaggler @Triskelion's implementation in Vowel Wabbit. (Many Thanks to @Triskelion)

* For the training phase, I have tried various models: 

 - Gradient Boosting Machine: gbm in R, xgboost from Tianqi Chen (https://github.com/tqchen/xgboost), GradientBoostingClassifier in scikit-learn. Among these, I have little luck with scikit-learn's implementation. It scores around 0.605. Other methods, even with small depth and small number of trees tend to overfit.
 - Random Forest: randomForest in R, RandomForestClassifier and RandomForestRegressor in scikit-learn. For RandomForestRegressor, I regressed to the repeattrips variable instead of the repeater factor. RandomForestClassifier seems work best with score around 0.602.
 - GLM: glmnet in R. This linear model turns out to work best with score around 0.608~0.609.


## Requirement

- R: glmnet, data.table, bit64, caTools

- Python: numpy, pandas, scikit-learn
  
  
## Instructions

* download data from the competition website: http://www.kaggle.com/c/acquire-valued-shoppers-challenge/data and put all the data into ./Data dir
 - ./Data/transactions.csv
 - ./Data/trainHistory.csv
 - ./Data/testHistory.csv
 - ./Data/offers.csv
 
* put all the code into ./ dir:
 - ./R/...
 - ./Python/...
 
* run ./Python/generate_features.py to generate the features

* run ./Python/train_gbm.py to train GradientBoostingClassifier (score around 0.605)

* run ./R/train_glm.R to train unbagging version of glm (score around 0.608~0.609)

* run ./R/train_glm_bagging.R to train bagging version of glm (score around 0.608~0.609)