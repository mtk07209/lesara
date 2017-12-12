#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:12:52 2017

@author: talhakhan
"""


import numpy as np
import sys
import os
import shutil
import time
import traceback

from flask import Flask, request, jsonify

from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import statsmodels.formula.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import mean_squared_error
from sklearn import metrics

###############################################################################
# Parameters for training Random Forest Classifier
#
TRAIN_PARTITION = 0.7       # train/test partition ratio
FOREST_SIZE = 5           # number of trees in forest
CRITERION = 'mse'          # criterion for splits (default='gini')
MAX_FEATURES = 'auto'       # max number of features to consider during split
VERBOSE = False             # verbose setting during forest construction




###############################################################################
# Evaluation

def dataset_statistics(dataset):
    """
    Basic statistics of the dataset
    :param dataset: Pandas dataframe
    :return: None, print the basic statistics of the dataset
    """
    print (dataset.describe())

#
def evaluate(model, x_test, y_test,y_predict):
  """
  Prints a summary of results for multiclass classification using a RFC.
  Also prints feature importances for the RFC.
  """
  
  
  print( "================================================================================")
  print ("Summary of Results:")
  
  print ("Forest Size :" , FOREST_SIZE)
 
  print("Accuracy Score: ",accuracy_score(y_test,y_predict))
  print("Mse: ",mean_squared_error(y_test,y_predict))
  #average_precision = average_precision_score(y_test, y_predict)
  #print(average_precision)
  
  #fpr, tpr, thresholds = roc_curve(y_test, y_predict, pos_label=2)
  fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=2)
  print("auc",metrics.auc(fpr, tpr))
   
   
 
  #print ("ROC : ", roc_curve(y_test,y_predict))
  #print ("AUC : ", auc(y_test,y_predict,reorder=False))
  

  
  print ("================================================================================")
  #print(average_precision=average_precision_score(Y_test, y_predict))
  #print("Average precision-recall score:", average_precision)
  print()
  print ("================================================================================")
  print( "Feature importance for Random Forest Classifier:\n")
  names=['client_id','host_name','page_path','click_time']
  print (sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), names), reverse=True))
 
  print ("================================================================================")
  print ("Done with evaluation")
  return None




###############################################################################
if __name__=="__main__":
    dataset=pd.read_csv("./train.csv")
    print(dataset_statistics(dataset))
    test_set=pd.read_csv("./test.csv")
    numeric=dataset[['click_time']]
    categorical=dataset[['host_name','page_path','client_id']]
    Y=dataset['gender']
    labelencoder_X=LabelEncoder()
    x2=categorical.apply(LabelEncoder().fit_transform)
    finaldf=pd.concat((x2,numeric),axis=1)
    imputer= Imputer(missing_values='NaN', strategy= 'median', axis = 0)
    imputer=imputer.fit(finaldf)
    X=imputer.transform(finaldf)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=TRAIN_PARTITION, random_state=0)
    sc_X=StandardScaler()
    X_train=sc_X.fit_transform(X_train)
    X_test=sc_X.fit_transform(X_test)
    #X_test=sc_X.fit_transform(X_test)
    regressor= RandomForestRegressor(n_estimators=FOREST_SIZE,criterion=CRITERION,max_features=MAX_FEATURES,verbose=VERBOSE)
    regressor.fit(X_train,Y_train)
        
    y_predict=regressor.predict(X_test)
    y_predict=y_predict.round()
    cm=confusion_matrix(Y_test,y_predict)
    
    plt.scatter(Y_test,Y_test, color='red')
    plt.scatter(Y_test,y_predict, color='blue')
    plt.xlabel('Actual Gender')
    plt.ylabel('Predicted gender')
    plt.show()
    # Evaluation
    evaluate(regressor, X_test, Y_test, y_predict)
    
   
    #transforming test set
    numeric1=test_set[['click_time']]
    categorical1=test_set[['host_name','page_path','client_id']]
    x3=categorical1.apply(LabelEncoder().fit_transform)
    finaldf1=pd.concat((x3,numeric1),axis=1)
    imputer= Imputer(missing_values='NaN', strategy= 'median', axis = 0)
    imputer=imputer.fit(finaldf1)
    X1=imputer.transform(finaldf1)
    X1=sc_X.fit_transform(X1)
    ypred=regressor.predict(X1)
    test_set['Gender']=ypred
    test_set.to_csv('~/Documents/lesara/Pred_test.csv')












