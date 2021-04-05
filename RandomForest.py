import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from fomlads.evaluate.eval_classification import f1_score
from sklearn.ensemble import RandomForestClassifier
import sys
import pickle

##run using python RandomForesy.py dataset drop_column_list category_label_list train_test_split_ratio useBalanced_weight
#  dataset: directory to data set
#  drop_column_list: no space list of column name separated by comma 
#  category_label_list: no space list of column name separated by comma fo variable that needs to be encoded into one hot vector
#  train_test_split_ratio: 0<x<1 the ratio for trainigand spliting data
#  useBalanced_weight: Boolean for using a balanced class weight
#       This is due to dataset has a imbalanced ratio of classes 
#  this will result in traning modals and providing some testing metrics on the model
#  the trained model will be saved as RF_model.object for future use
#
## example
# python RandomForest.py Churn_Modelling.csv RowNumber,CustomerId,Surname Geography,Gender 0.2 True 




def cf_matrix(targets, predicts):
    true_positives = np.sum((predicts == 1) & (targets == 1))
    true_negative = np.sum((predicts == 0) & (targets == 0))
    false_negative = np.sum((predicts == 0) & (targets == 1))
    false_positive = np.sum((predicts == 1) & (targets == 0))
    return np.array([[true_negative, false_positive],[false_negative, true_positives]])

#Fixing all random state
np.random.seed(123)

input_data = pd.read_csv(sys.argv[1])
input_data.head()

print("Checking Data")
print('\nNull Values \n{}'.format(input_data.isnull().sum()))

print('\nDuplicated values {}'.format(input_data.duplicated().sum()))

coluumn_to_drop = list(sys.argv[2].split(','))
print("Dropping column:", coluumn_to_drop)

input_data = input_data.drop(coluumn_to_drop, axis=1)

word_label = list(sys.argv[3].split(','))
print("Category Label:", word_label)
removing = []
for column in word_label:
    if(input_data[column].dtype == np.str or input_data[column].dtype == np.object):
        for cat in input_data[column].unique():
            input_data[column+'_'+cat] = np.where(input_data[column] == cat, 1, 0)
        removing.append(column)
input_data = input_data.drop(removing, axis = 1)

print("Data after pre-processing")
print(input_data.head())

#Split Data
from fomlads.model.classification import split_train_test
train_set, test_set = split_train_test(input_data, test_ratio=float(sys.argv[4]))
X_train = train_set.drop(['Exited'], axis=1)
y_train = train_set['Exited']
X_test = test_set.drop(['Exited'], axis=1)
y_test = test_set['Exited']

print("Building Classifier")
if sys.argv[5] == "True":
    print("building classifier with balanced weight")
    rf_Classifier = RandomForestClassifier(n_estimators=10, class_weight="balanced", random_state=123)
    rf_Classifier.fit(X_train, y_train)
    
else:
    print("Not using balanced weight")
    rf_Classifier = RandomForestClassifier(n_estimators=10, random_state=123)
    rf_Classifier.fit(X_train, y_train)

pickle.dump(rf_Classifier, open("RF_model.object", "wb+"))

y_pred = rf_Classifier.predict(X_test)

print("Metrics:")
print("Confusion Matrix")
conf_matrix = cf_matrix(y_test, y_pred)
print(conf_matrix)
print("F1_Score")
print(f1_score(y_test, y_pred))
