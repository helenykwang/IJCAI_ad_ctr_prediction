#coding:utf-8

import pandas as pd
import numpy as np
import time
from sklearn import metrics
###################################################
# 

feat_set = ["context_page_id","shop_id","shop_review_num_level","shop_review_positive_rate",\
            "shop_star_level","shop_score_service","shop_score_delivery","shop_score_description",\
            "item_id","ctx_hour","ctx_minitue","item_category_1",\
            "item_property_0","item_property_1","item_property_2","item_property_3","item_property_4","item_property_5",\
            "predict_category_property_0","predict_category_property_1","predict_category_property_2","predict_category_property_3","predict_category_property_4","predict_category_property_5",\
            "item_brand_id","item_city_id","user_gender_id","user_occupation_id"]

x_remain = ["is_busi_time", "is_good_sevice", "is_good_desc",  "is_good_review", \
            "predict_category_property_size","predict_category_property_null_size","item_price_level","item_sales_level","item_collected_level","item_pv_level",\
            "viewed_items_count", "viewed_category_count", "user_age_level","user_star_level"]

    
print('train')
train = pd.read_csv('./data/20180331/trainset0331.txt',sep=" ")

val = train[train['ctr_date']>='0923'] 
train = train[train['ctr_date']<='0922']



print('test')
test_a = pd.read_csv('./data/20180331/testset0331.txt',sep=" ")
# test_a = pd.read_csv('./data/20180331/testset0331.txt',sep=" ",nrows = 300)

test_index = test_a.pop("instance_id")

y_train = train.pop('is_trade')
y_val = val.pop('is_trade')


print('preprocessing...')

from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from sklearn.linear_model import LogisticRegression

lb = LabelEncoder()

X_train,  X_test, X_val = pd.DataFrame(index = train.index), pd.DataFrame(index = test_a.index), pd.DataFrame(index = val.index)

for i,feat in enumerate(feat_set):
    lb.fit_transform((list(train[feat])+list(val[feat])+list(test_a[feat])))
    X_train[feat] = lb.transform(train[feat])
    X_test[feat] = lb.transform(test_a[feat])
    X_val[feat] = lb.transform(val[feat])


X_train = pd.concat([X_train, train[x_remain]], axis=1)
X_test = pd.concat([X_test, test_a[x_remain]], axis=1)
X_val = pd.concat([X_val, val[x_remain]], axis=1)

print "X_train:", X_train.shape,"X_val: ",X_val.shape,"X_test: ",X_test.shape
print "y_train: ", y_train.shape, "y_val: ", y_val.shape


import lightgbm as lgb
print('Start training...')
gbm = lgb.LGBMRegressor(objective='binary',
                        num_leaves=64,
                        learning_rate = 0.01,
                        colsample_bytree = 0.8,
                        subsample = 0.8,
                        n_estimators=1000)

gbm.fit(X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        early_stopping_rounds=50)

print('Start predicting...')

# predict

y_pred_1 = gbm.predict(X_val, num_iteration=gbm.best_iteration_)

y_sub_1 = gbm.predict(X_test, num_iteration=gbm.best_iteration_)


