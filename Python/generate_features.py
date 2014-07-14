# -*- coding: UTF-8 -*-

"""
Code provided here for constructing features for:
Kaggle's Acquire Valued Shoppers Challenge
http://www.kaggle.com/c/acquire-valued-shoppers-challenge

It is based on Kaggle user @Triskelion's code:
http://www.kaggle.com/c/acquire-valued-shoppers-challenge/forums/t/7688/
feature-engineering-and-beat-the-benchmark-0-59347

Python version: 2.7.6
Version: 1.0 at Jun 29 2014
Author: Chenglong Chen < yr@Kaggle >
Email: c.chenglong@gmail.com
"""


import os
import gc
import sys
import csv
import math
import cProfile
from datetime import datetime
from collections import defaultdict


######################
## Helper functions ##
######################
#### This function computes the mean value
# or simply use np.mean
def func_mean(lst):
    return sum(lst) / float(len(lst))
    
    
#### This function computes the median value
# or simply use np.median
def func_median(lst):
    even = (0 if len(lst) % 2 else 1) + 1
    half = (len(lst) - 1) // 2
    return sum(sorted(lst)[half:half + even]) / float(even)

    
#### This function reduces the whole transcation record data
def func_reduce_data(path_to_offers, path_to_transactions, path_to_reduced):
    start = datetime.now()
    #get all categories and comps on offer in a dict
    offers_category = {}
    offers_dept = {}
    offers_company = {}
    offers_brand = {}
    for e, line in enumerate( open(path_to_offers) ):
        category = line.split(',')[1]
        if e == 0:
            dept = 'dept'
        else:
            dept = str(int(math.floor(float(category)/100.0)))
        company = line.split(',')[3]
        brand = line.split(',')[5]
        offers_company[ company ] = 1
        offers_category[ category ] = 1
        offers_dept[ dept ] = 1
        offers_brand[ brand ] = 1
    #open output file
    with open(path_to_reduced, 'wb') as outfile:
        #go through transactions file and reduce
        reduced = 0
        for e, line in enumerate( open(path_to_transactions) ):
            if e == 0:
                outfile.write( line ) #print header
            else:
                #only write when if category in offers dict
                category = line.split(',')[3]
                dept = str(int(math.floor(float(category)/100.0)))
                company = line.split(',')[4]
                brand = line.split(',')[5]
                # sinace dept is an aggregrated of category, we can only compare dept
                if offers_dept.has_key(dept) or offers_company.has_key(company) or offers_brand.has_key(brand):
                    outfile.write( line )
                    reduced += 1
            #progress
            if e % 5000000 == 0:
                print e, reduced, datetime.now() - start
    print e, reduced, datetime.now() - start


#### This function computes the number of days between two dates
def func_diff_days(s1,s2):
    date_format = '%Y-%m-%d'
    a = datetime.strptime(s1, date_format)
    b = datetime.strptime(s2, date_format)
    delta = b - a
    return delta.days


#### This function generates features from the transactions record
def func_generate_features(path_to_train, path_to_test, path_to_offers, path_to_transactions,
                           date_diff_days_thresh, path_to_out_train, path_to_out_test, DEBUG):
    # keep a dictionary with the offer data
    offers = {}
    with open( path_to_offers ) as offer_file:
        for e, line in enumerate( offer_file ):
            row = line.strip().split(',')
            offers[ row[0] ] = row
    
    # keep a dictionary with the train history
    train_history = {}
    with open( path_to_train ) as train_history_file:
        for e, line in enumerate( train_history_file ):
            if e > 0:
                row = line.strip().split(',')
                train_history[ row[0] ] = row
                
    # keep a dictionary with the test history
    test_history = {}
    with open( path_to_test ) as test_history_file:
        for e, line in enumerate( test_history_file ):
            if e > 0:
                row = line.strip().split(',')
                test_history[ row[0] ] = row
    
    # features for individual shopper id
    features_list_train = []
    features_list_test = []
    done_header_row_train = False
    done_header_row_test = False
    if DEBUG:
        cache_row_size = 100
    else:
        cache_row_size = 10000
    header_names = set()
    # overall features for the offer/company/category/dept/brand
    features_overall = defaultdict(float)
    
    # open two temporary files for saving features
    path_to_out_train_shopper = path_to_out_train[:-4] + '_shopper.csv'
    path_to_out_test_shopper = path_to_out_test[:-4] + '_shopper.csv'
    out_train_shopper = open( path_to_out_train_shopper, 'wb' )
    out_test_shopper = open( path_to_out_test_shopper, 'wb' )
    with open( path_to_transactions ) as transactions_file:
        #iterate through reduced dataset 
        last_id = 0
        features = defaultdict(float)
        for e, line in enumerate( transactions_file ):
            if e > 500000 and DEBUG:
                break
            elif e > 0: #skip header
                # poor man's csv reader
                row = line.strip().split(',')
                
                # extract info from this row
                row_id = row[0]
                row_company = row[4]
                row_category = row[3]
                row_brand = row[5]
                row_dept = math.floor(float(row_category)/100.0)
                row_date = row[6]
                
                size = float( row[7] )
                quantity = float( row[9] )
                amount = float( row[10] )
                TYPES_VALUES = { 't': 1.0, 's': size, 'q': quantity, 'a': amount }
                
                ##############################
                ## get the overall features ##
                ##############################
                KEYS = ['overall_bought_company{0}'.format(row_company),
                        'overall_bought_category{0}'.format(row_category),
                        'overall_bought_brand{0}'.format(row_brand),
                        'overall_bought_dept{0}'.format(row_dept),
                        'overall_bought_company{0}_category{1}'.format(row_company,row_category),
                        'overall_bought_company{0}_brand{1}'.format(row_company,row_brand),
                        'overall_bought_category{0}_brand{1}'.format(row_category,row_brand),
                        'overall_bought_company{0}_dept{1}'.format(row_company,row_dept),
                        'overall_bought_dept{0}_brand{1}'.format(row_dept,row_brand),
                        'overall_bought_company{0}_category{1}_brand{2}'.format(row_company,row_category,row_brand),
                        'overall_bought_company{0}_dept{1}_brand{2}'.format(row_company,row_dept,row_brand)
                        ]
                        
                for k in KEYS:
                    for tt, vv in TYPES_VALUES.items():
                        features_overall['{0}_{1}'.format(k,tt)] += vv
                    
                
                #########################################
                ## get the feature for each shopper id ##
                #########################################
                # deal with the last row of this shopper id in the transaction record
                if last_id != row_id and e != 1:
                    
                    ####
                    D = {
                          'shopper_never_bought_offer_company': features.has_key('shopper_bought_offer_company_t'),
                          'shopper_never_bought_offer_category': features.has_key('shopper_bought_offer_category_t'),
                          'shopper_never_bought_offer_brand': features.has_key('shopper_bought_offer_brand_t'),
                          'shopper_never_bought_offer_dept': features.has_key('shopper_bought_offer_dept_t'),
                          'shopper_never_bought_offer_company_category': features.has_key('shopper_bought_offer_company_category_t'),
                          'shopper_never_bought_offer_company_brand': features.has_key('shopper_bought_offer_company_brand_t'),
                          'shopper_never_bought_offer_category_brand': features.has_key('shopper_bought_offer_category_brand_t'),
                          'shopper_never_bought_offer_company_dept': features.has_key('shopper_bought_offer_company_dept_t'),
                          'shopper_never_bought_offer_dept_brand': features.has_key('shopper_bought_offer_dept_brand_t'),
                          'shopper_never_bought_offer_company_category_brand': features.has_key('shopper_bought_offer_company_category_brand_t'),
                          'shopper_never_bought_offer_company_dept_brand': features.has_key('shopper_bought_offer_company_dept_brand_t')
                    }
                    for k, v in D.items():
                        if not v:
                            features[k] = 1
                        else:
                            features[k] = 0
                            
                    ####
                    features['shopper_bought_size_median'] = func_median( features['shopper_bought_size_median'] )
                    features['shopper_bought_quantity_median'] = func_median( features['shopper_bought_quantity_median'] )
                    features['shopper_bought_amount_median'] = func_median( features['shopper_bought_amount_median'] )
                    
                    ####
                    this_company = float(features['shopper_bought_company_num'][features['company']])
                    all_company = float(sum(features['shopper_bought_company_num'].values()))
                    features['shopper_bought_offer_company_ratio'] = this_company / all_company
                    features['shopper_bought_company_median_time'] = func_median(features['shopper_bought_company_num'].values())
                    features['shopper_bought_company_num'] = len(features['shopper_bought_company_num'].keys())
                    
                    this_category = float(features['shopper_bought_category_num'][features['category']])
                    all_category = float(sum(features['shopper_bought_category_num'].values()))
                    features['shopper_bought_offer_category_ratio'] = this_category / all_category
                    features['shopper_bought_category_median_time'] = func_median(features['shopper_bought_category_num'].values())
                    features['shopper_bought_category_num'] = len(features['shopper_bought_category_num'].keys())
                    
                    this_dept = float(features['shopper_bought_dept_num'][features['dept']])
                    all_dept = float(sum(features['shopper_bought_dept_num'].values()))
                    features['shopper_bought_offer_dept_ratio'] = this_dept / all_dept
                    features['shopper_bought_dept_median_time'] = func_median(features['shopper_bought_dept_num'].values())
                    features['shopper_bought_dept_num'] = len(features['shopper_bought_dept_num'].keys())
                    
                    this_brand = float(features['shopper_bought_brand_num'][features['brand']])
                    all_brand = float(sum(features['shopper_bought_brand_num'].values()))
                    features['shopper_bought_offer_brand_ratio'] = this_brand / all_brand
                    features['shopper_bought_brand_median_time'] = func_median(features['shopper_bought_brand_num'].values())
                    features['shopper_bought_brand_num'] = len(features['shopper_bought_brand_num'].keys())
                    
                    ####
                    features['shopper_bought_date_median_time'] = func_median(features['shopper_bought_date_num'].values())
                    date_ = features['shopper_bought_date_num'].keys()
                    features['shopper_bought_date_num'] = len(date_)

                    if len(date_) == 1:
                        date_gap = [0] # may use an indiator
                    else:
                        date_gap = []
                        date_ = sorted(date_)
                        for i in xrange(len(date_)-1):
                            date_gap.append( func_diff_days(date_[i], date_[i+1]) )
                    features['shopper_bought_date_median_gap'] = func_median(date_gap)
                    
                    ####
                    if features['repeater'] == 0.5:
                        # not ready to write
                        if done_header_row_train == False:
                            features_list_test.append( features )
                            header_names.update( features.keys() )
                        else:
                            if done_header_row_test == False:
                                features_list_test.append( features )
                                # insert header row
                                features_list_test.insert(0, header_row)
                                #### write to csv file
                                writer_test = csv.DictWriter(out_test_shopper, header_names)
                                writer_test.writerows( features_list_test )
                                # reset & clean up
                                done_header_row_test = True
                                del features_list_test # del
                                gc.collect() # gc
                            else:
                                writer_test.writerow( features )
                    else:
                        # deal with header row
                        if done_header_row_train == False:
                            features_list_train.append( features )
                            header_names.update( features.keys() )
                            if len(features_list_train) == cache_row_size:
                                # Move some keys to the front
                                header_names = list(header_names)
                                key_to_adjust = ['id', 'offer', 'repeater', 'repeattrips', 'offer_value',
                                                 'offer_date', 'offer_days', 'offer_mday', 'offer_mweek', 'offer_weekday', 
                                                 'company', 'category', 'dept', 'brand', 'market', 'chain' ]
                                for k in key_to_adjust:
                                    header_names.remove(k)
                                for k in key_to_adjust[::-1]:
                                    header_names.insert(0, k)
                                # header row
                                #header_row = {key:key for key in header_names}
                                # use the following for compatibility in Python 2.6...
                                header_row = {}
                                for key in header_names:
                                    header_row[key] = key
                                # insert header row
                                features_list_train.insert(0, header_row)
                                #### write to csv file
                                writer_train = csv.DictWriter(out_train_shopper, header_names)
                                writer_train.writerows( features_list_train )
                                # reset & clean up
                                done_header_row_train = True
                                del features_list_train # del
                                gc.collect() # gc
                        else:
                            writer_train.writerow( features )
                    
                    #reset features
                    features = defaultdict(float)
                    
                # generate features from transaction record
                # check if we have a test sample or train sample
                if train_history.has_key(row_id) or test_history.has_key(row_id):
                    features['id'] = row_id
                    # generate label and history
                    if train_history.has_key(row_id):
                        if train_history[row_id][5] == 't':
                            features['repeater'] = 1
                        else:
                            features['repeater'] = 0
                        features['repeattrips'] = train_history[row_id][4]                        
                        history = train_history[row_id]
                    else:
                        features['repeater'] = 0.5
                        features['repeattrips'] = 0.5
                        history = test_history[row_id]
                        
                    ####
                    offer_company = offers[ history[2] ][3]
                    offer_category = offers[ history[2] ][1]
                    offer_brand = offers[ history[2] ][5]
                    offer_dept = math.floor(float(offer_category)/100.0)
                    
                    date_diff_days = func_diff_days(row_date, history[-1])
                    
                    ####
                    features['company'] = offer_company
                    features['category'] = offer_category
                    features['brand'] = offer_brand
                    features['dept'] = offer_dept
                    
                    features['chain'] = history[1]
                    features['offer'] = history[2]
                    features['market'] = history[3]
                    features['offer_value'] = offers[ history[2] ][4]
                    # quantity in training set are all 1 proving usefulness
                    #features['offer_quantity'] = offers[ history[2] ][2]                    
                        
                    # offer date
                    offer_date = history[-1]
                    date_format = '%Y-%m-%d'
                    offer_date = datetime.strptime(offer_date, date_format)
                    features['offer_date'] = history[-1]
                    # add days from the based_date '2013-03-01'
                    base_date = '2013-03-01'
                    base_date = datetime.strptime(base_date, date_format)
                    features['offer_days'] = (offer_date - base_date).days
                    # and day in month
                    features['offer_mday'] = offer_date.day
                    # add week in month
                    features['offer_mweek'] = math.floor(offer_date.day/4.0)
                    # add weekday in month
                    features['offer_weekday'] = offer_date.weekday()

                    
                    features['shopper_bought_time_total'] += 1.0                    
                    features['shopper_bought_size_total'] += size
                    features['shopper_bought_quantity_total'] += quantity
                    features['shopper_bought_amount_total'] += amount
                    

                    ####
                    if not features.has_key('shopper_bought_size_median'):
                        features['shopper_bought_size_median'] = [ size ]
                    else:
                        features['shopper_bought_size_median'].append( size )
                        
                    if not features.has_key('shopper_bought_quantity_median'):
                        features['shopper_bought_quantity_median'] = [ quantity ]
                    else:
                        features['shopper_bought_quantity_median'].append( quantity )
                        
                    if not features.has_key('shopper_bought_amount_median'):
                        features['shopper_bought_amount_median'] = [ amount ]
                    else:
                        features['shopper_bought_amount_median'].append( amount )

                    ####
                    if not features.has_key('shopper_bought_company_num'):
                        features['shopper_bought_company_num'] = defaultdict(float)
                    features['shopper_bought_company_num'][row_company] += 1.0
                        
                    if not features.has_key('shopper_bought_category_num'):
                        features['shopper_bought_category_num'] = defaultdict(float)
                    features['shopper_bought_category_num'][row_category] += 1.0
                    
                    if not features.has_key('shopper_bought_dept_num'):
                        features['shopper_bought_dept_num'] = defaultdict(float)
                    features['shopper_bought_dept_num'][row_dept] += 1.0
                    
                    if not features.has_key('shopper_bought_brand_num'):
                        features['shopper_bought_brand_num'] = defaultdict(float)
                    features['shopper_bought_brand_num'][row_brand] += 1.0

                    # count date
                    if not features.has_key('shopper_bought_date_num'):
                        features['shopper_bought_date_num'] = defaultdict(float)
                    features['shopper_bought_date_num'][row_date] += 1.0
                    
                    ####
                    D = {
                          'shopper_returned': amount < 0 or size < 0 or quantity < 0,
                          'shopper_bought_offer_company': offer_company == row_company,
                          'shopper_bought_offer_category': offer_category == row_category,
                          'shopper_bought_offer_dept': offer_dept == row_dept,
                          'shopper_bought_offer_brand': offer_brand == row_brand,
                          'shopper_bought_offer_company_category': offer_company == row_company and offer_category == row_category,
                          'shopper_bought_offer_company_brand': offer_company == row_company and offer_brand == row_brand,
                          'shopper_bought_offer_category_brand': offer_category == row_category and offer_brand == row_brand,
                          'shopper_bought_offer_company_dept': offer_company == row_company and offer_dept == row_dept,
                          'shopper_bought_offer_dept_brand': offer_dept == row_dept and offer_brand == row_brand,
                          'shopper_bought_offer_company_category_brand': offer_company == row_company and offer_category == row_category and offer_brand == row_brand,
                          'shopper_bought_offer_company_dept_brand': offer_company == row_company and offer_dept == row_dept and offer_brand == row_brand
                    }

                    for k, v in D.items():
                        if v == 1:
                            for tt, vv in TYPES_VALUES.items():
                                if k == 'shopper_returned':
                                    vv = abs(vv) # convert negative to positive
                                features['{0}_{1}'.format(k,tt)] += vv
                            for thresh in date_diff_days_thresh:
                                if date_diff_days < thresh:
                                    for tt, vv in TYPES_VALUES.items():
                                        if k == 'shopper_returned':
                                            vv = abs(vv) # convert negative to positive
                                        features['{0}_{1}_{2}'.format(k,tt,thresh)] += vv
                                
                last_id = row_id
                if e % 100000 == 0:
                    print(e)
                                   
    out_train_shopper.close()
    out_test_shopper.close()
    
    #######################################
    ## combine to get the final features ##
    #######################################
    K = [
          'overall_bought_company',
          'overall_bought_category',
          'overall_bought_brand',
          'overall_bought_dept',
          'overall_bought_company_category',
          'overall_bought_company_brand',
          'overall_bought_category_brand',
          'overall_bought_company_dept',
          'overall_bought_dept_brand',
          'overall_bought_company_category_brand',
          'overall_bought_company_dept_brand'
    ]
    KEYS = []
    for k in K:
        for tt in TYPES_VALUES.keys():
            KEYS.append( '{0}_{1}'.format(k, tt) )
    header_names = header_names + KEYS
    #header_row = {key:key for key in header_names}
    # use the following for  compatibility in Python 2.6...
    header_row = {}
    for key in header_names:
        header_row[key] = key
    
                    
    # training data
    out_train_shopper = open(path_to_out_train_shopper, 'rb')
    with open(path_to_out_train, 'wb') as out_train:
        writer_train = csv.DictWriter(out_train, header_names)
        writer_train.writerow( header_row )
        for e, line in enumerate( out_train_shopper ):
            if e > 0: #skip header
                f = line.split(',')
                #features = {k:v for k, v in zip(header_names, f)}
                # use the following for  compatibility in Python 2.6...
                features = {}
                for k, v in zip(header_names, f):
                    features[k] = v
                features = func_combine_shopper_overall_features(features, features_overall)
                writer_train.writerow( features )
    out_train_shopper.close()

    
    # testing data
    out_test_shopper = open(path_to_out_test_shopper, 'rb')
    with open(path_to_out_test, 'wb') as out_test:
        writer_test = csv.DictWriter(out_test, header_names)
        writer_test.writerow( header_row )
        for e, line in enumerate( out_test_shopper ):
            if e > 0: #skip header
                f = line.split(',')
                #features = {k:v for k, v in zip(header_names, f)}
                # use the following for  compatibility in Python 2.6...
                features = {}
                for k, v in zip(header_names, f):
                    features[k] = v
                features = func_combine_shopper_overall_features(features, features_overall)
                writer_test.writerow( features )
    out_test_shopper.close()
    
    # remove temporary file
    #os.remove(path_to_out_train_shopper)
    #os.remove(path_to_out_test_shopper)
     
     
#### This function combines shopper features with overall features     
def func_combine_shopper_overall_features(features, features_overall):
    company = features['company']
    category = features['category']
    brand = features['brand']
    dept = features['dept']
    D = {
          'overall_bought_company': 'overall_bought_company{0}'.format(company),
          'overall_bought_category': 'overall_bought_category{0}'.format(category),
          'overall_bought_brand':'overall_bought_brand{0}'.format(brand),
          'overall_bought_dept':'overall_bought_dept{0}'.format(dept),
          'overall_bought_company_category':'overall_bought_company{0}_category{1}'.format(company,category),
          'overall_bought_company_brand':'overall_bought_company{0}_brand{1}'.format(company,brand),
          'overall_bought_category_brand':'overall_bought_category{0}_brand{1}'.format(category,brand),
          'overall_bought_company_dept':'overall_bought_company{0}_dept{1}'.format(company,dept),
          'overall_bought_dept_brand':'overall_bought_dept{0}_brand{1}'.format(dept,brand),
          'overall_bought_company_category_brand':'overall_bought_company{0}_category{1}_brand{2}'.format(company,category,brand),
          'overall_bought_company_dept_brand':'overall_bought_company{0}_dept{1}_brand{2}'.format(company,dept,brand)
    }
    TYPES = [ 't', 's', 'q', 'a' ]
    for k,v in D.items():
        for tt in TYPES:
            features['{0}_{1}'.format(k,tt)] = features_overall['{0}_{1}'.format(v,tt)]
            
    return features
   
   
##########
## Main ##
##########
   
if __name__ == '__main__':
    
    DEBUG = False
    
    # for my own use on linux cluster
    if sys.platform == 'linux' or sys.platform == 'linux2':
        home_path = '/home/chchengl/Acquire'
    elif sys.platform == 'win32':
        home_path = '..'
    
  
    path_to_offers = home_path + '/Data/offers.csv'
    path_to_transactions = home_path + '/Data/transactions.csv'
    path_to_train = home_path + '/Data/trainHistory.csv'
    path_to_test = home_path + '/Data/testHistory.csv'
    
    # will be created
    path_to_reduced = home_path + '/Data/reduced.csv' 
    path_to_out_train = home_path + '/Data/train_Jun27.csv'
    path_to_out_test = home_path + '/Data/test_Jun27.csv'
    
    date_diff_days_thresh = range(30,361,30)
    date_diff_days_thresh = [ 30, 60, 90, 120, 150, 180 ]
    date_diff_days_thresh = [ 10, 20, 30, 45, 60, 90, 120, 180, 240, 360 ]
    
    # if you want to reduce the transactions data
    #func_reduce_data(path_to_offers, path_to_transactions, path_to_reduced)
    
    if DEBUG:
        cProfile.run('func_generate_features(path_to_train, path_to_test, path_to_offers, path_to_transactions,\
                          date_diff_days_thresh, path_to_out_train, path_to_out_test, DEBUG)')
    else:
        func_generate_features(path_to_train, path_to_test, path_to_offers, path_to_transactions,
                               date_diff_days_thresh, path_to_out_train, path_to_out_test, DEBUG)
