# -*- coding: utf-8 -*-
"""
dj
"""
import argparse
from sklearn import metrics
import numpy as np
import h5py
import warnings
warnings.filterwarnings("ignore")
'''
parse = argparse.ArgumentParser()
parse.add_argument(
    '--model',
    nargs='+',
    help='inceptionv3, vgg, densenet121',
    default=['vgg', 'inceptionv3', 'densenet121'])
opt = parse.parse_args()
print(opt)
root = '/home/cc/Desktop/kaggle_dog_vs_cat/model/'
train_list = ['train_feature_{}.hd5f'.format(i) for i in opt.model]
val_list = ['val_feature_{}.hd5f'.format(i) for i in opt.model]
def read_data():

    label_file = h5py.File(train_list[0], 'r')
    label = np.array(label_file.get('label'))
    nSamples = len(label)
    temp_dataset = np.empty([nSamples,0])
    for file in train_list:
        h5_file = h5py.File(file, 'r')
        dataset = np.array(h5_file.get('data'))
        temp_dataset = np.concatenate([temp_dataset, dataset], axis = 1)
    return temp_dataset, label

x,y=read_data()


val_list = ['val_feature_{}.hd5f'.format(i) for i in opt.model]
def read_data1():

    label_file = h5py.File(val_list[0], 'r')
    label = np.array(label_file.get('label'))
    nSamples = len(label)
    temp_dataset = np.empty([nSamples,0])
    for file in val_list:
        h5_file = h5py.File(file, 'r')
        dataset = np.array(h5_file.get('data'))
        temp_dataset = np.concatenate([temp_dataset, dataset], axis = 1)
    return temp_dataset, label
#print(x,y)

x_test,y_test=read_data1()
import xgboost
import numpy as np

xgboost=xgboost.XGBClassifier()
xgboost.fit(x,y)
pred=xgboost.predict(np.array(x_test))
print('model precision:\n',metrics.accuracy_score(y_test,pred))
print('model parameters:\n',metrics.classification_report(y_test,pred))
'''
def read_data(path):   
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label
x,y=read_data('/home/cc/Desktop/kaggle_dog_vs_cat/model/train_feature_vgg.hd5f')
x_test,y_test=read_data('/home/cc/Desktop/kaggle_dog_vs_cat/model/val_feature_vgg.hd5f')   
'''
import xgboost
import numpy as np

xgboost=xgboost.XGBClassifier()
xgboost.fit(x,y)
pred=xgboost.predict(np.array(x_test))
print('model precision:\n',metrics.accuracy_score(y_test,pred))
print('model parameters:\n',metrics.classification_report(y_test,pred))
'''
import numpy as np
import lightgbm as lgb
print("LGB test")
clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=55, reg_alpha=0.0, reg_lambda=1,
        max_depth=15, n_estimators=6000, objective='binary',
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4
    )
clf.fit(x, y)
pred=clf.predict(np.array(x_test))
print('model precision:\n',metrics.accuracy_score(y_test,pred))
print('model parameters:\n',metrics.classification_report(y_test,pred))
print("starting first testing......")  
clf = lgb.LGBMClassifier(  
        boosting_type='gbdt', num_leaves=50, reg_alpha=0.0, reg_lambda=1,  
        max_depth=-1, n_estimators=1500, objective='binary',  
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,  
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=100  
    )  
clf.fit(x,y, eval_set=[(x, y)], eval_metric='auc',early_stopping_rounds=1000)  
pre1=clf.predict(x_test)  
print('model precision:\n',metrics.accuracy_score(y_test,pred))
print('model parameters:\n',metrics.classification_report(y_test,pred))















        
      
  


