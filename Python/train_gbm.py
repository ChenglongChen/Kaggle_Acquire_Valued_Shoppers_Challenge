# -*- coding: UTF-8 -*-

"""
Code provided here for training GBM for:
Kaggle's Acquire Valued Shoppers Challenge
http://www.kaggle.com/c/acquire-valued-shoppers-challenge

Python version: 2.7.6
Version: 1.0 at Jun 29 2014
Author: Chenglong Chen < yr@Kaggle >
Email: c.chenglong@gmail.com
"""


import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier


######################
## Helper functions ##
######################

#### This function converts categorical vars to numeric values
# You could also try OneHotEncoder in sklearn...
def categorical_to_numerical(X):
    if len(X.shape) == 1:
        _, XX = np.unique(X, return_inverse=True)
    elif len(X.shape) == 2:
        XX = np.zeros(X.shape)
        for i in np.arange(X.shape[1]):
            _, XX[:,i] = np.unique(X[:,i], return_inverse=True)
    else:
        raise(ValueError('Do not support array of shape > 2'))
    return XX
    
    
#### This function computes the AUC
def func_compute_AUC(labels, scores):
    '''
    Computes AUC of ROC curve.
    '''
    assert len(labels) == len(scores)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    return(roc_auc)
    

##########
## Main ##
##########
def main():
    
    # path to where the data lies
    data_path = '../Data/'
    # path to save csv file
    filePath_csv = '../Submission/GBM/'
    
    #### Load training features from csv file
    print 'Load features'
    train_feat_file = data_path + 'train_Jun27_shopper.csv'
    dfTrain = pd.read_csv(train_feat_file)
    dfTrain = dfTrain.fillna(0.0)
    numTrain = dfTrain.shape[0]
    
    #### Load testing features from csv file
    test_feat_file = data_path + 'test_Jun27_shopper.csv'
    dfTest = pd.read_csv(test_feat_file)
    dfTest = dfTest.fillna(0.0)
    ID_test = dfTest['id'].values
    
    # convert categorical variables
    categorical_names = [
    'offer', 'market', 'company', 'category', 'brand',
    'dept', 'offer_mday', 'offer_mweek', 'offer_weekday',
    ]
    df = pd.concat([dfTrain[categorical_names], dfTest[categorical_names]])
    df_c = categorical_to_numerical(df.values)
    dfTrain[categorical_names] = df_c[:numTrain,:]
    dfTest[categorical_names] = df_c[numTrain:,:]
    
    # get the names of predictors
    all_vars = dfTrain.columns
    unused = [
    'id', 'repeater', 'repeattrips', 'offer_date',
    'offer_mday', 'offer_days', 'offer_weekday'
    ]
    predictors = list(all_vars - unused)


    print 'Perform training-validation'
    offer_date = dfTrain['offer_date'].values

    train_fraction = 0.6

    order = offer_date.argsort()
    RankOrder = order.argsort()
    top = np.int32(np.floor(train_fraction * len(offer_date)))
    thresh = offer_date[RankOrder==top][0]
    thresh = '2013-04-01'

    train_idx = np.where(offer_date < thresh)[0]
    valid_idx = np.where(offer_date >= thresh)[0]

    # train GBM
    # params for GBM
    GBM_ntree = 1000
    GBM_subsample = 0.5
    GBM_lr = 0.01
    random_seed = 2014    
    clf = GradientBoostingClassifier(n_estimators=GBM_ntree,
                                     subsample=GBM_subsample,
                                     learning_rate=GBM_lr,
                                     random_state=random_seed,
                                     verbose=3)
    
    clf.fit(dfTrain[predictors].values[train_idx,:], dfTrain['repeater'].values[train_idx])
    y_valid_pred = clf.predict_proba(dfTrain[predictors].values[valid_idx,:])[:,1]
    auc_valid = func_compute_AUC(dfTrain['repeater'].values[valid_idx], y_valid_pred)
    print 'Valid AUC: {}'.format(auc_valid)


    print 'Train the final model'
    clf.fit(dfTrain[predictors].values, dfTrain['repeater'].values)                      
    print 'Make prediction'
    p = clf.predict_proba(dfTrain[predictors].values)[:,1]
    
    print 'Create submission'
    test_history_file = data_path + 'testHistory.csv'
    dfTestHistory = pd.read_csv(test_history_file)
    allID = np.asarray(dfTestHistory.values[:,0], dtype=np.int64)
    idx = np.where(pd.DataFrame(allID).isin(ID_test))[0]

    y_test_prob = np.zeros((len(allID),))
    y_test_prob[idx] = p
    
    sub = dict()
    sub['id'] = allID
    sub['repeatProbability'] = y_test_prob
    sub = pd.DataFrame(sub)
    fileName = filePath_csv + 'GBM_[Feature_Jun27]_[Ntree{}]_[Bag{}]_[lr{}]_[AUC{}].csv'.format(GBM_ntree, GBM_subsample, GBM_lr, np.round(auc_valid,5))
    sub.to_csv(fileName, index = False)
    

if __name__ == '__main__':
    main()