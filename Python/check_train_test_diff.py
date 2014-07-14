# -*- coding: UTF-8 -*-

"""
Based on code from Zygmunt Zając <zygmunt@fastml.com>
"""

import pandas as pd
train = pd.read_csv( '../Data/trainHistory.csv' )
test = pd.read_csv( '../Data/testHistory.csv' )
for col in ( 'offer', 'chain', 'market', 'id' ):
    print '-----------------------------'
    print '{}:'.format(col)
    print "train in test:", sum( train[col].isin( test[col] ))
    print "test in train:", sum( test[col].isin( train[col] ))
print '-----------------------------'
    