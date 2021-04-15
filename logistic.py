import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain


import numpy as np 

from fomlads.data.external import import_for_classification
from fomlads.data.external import standard_scaler
from fomlads.plot.evaluations import plot_roc
from fomlads.plot.evaluations import plot_train_test_errors

from fomlads.evaluate.eval_logistic import test_parameter_logistic
from fomlads.evaluate.eval_classification import score, two_class_cf_matrix, roc , roc_auc, two_class_f1_score

from fomlads.model.classification import logistic_regression_fit
from fomlads.model.classification import logistic_regression_predict, logistic_regression_prediction_probs
from fomlads.model.classification import split_train_test
    

def fit_evaluate_logistic(train_inputs, train_targets, test_inputs, test_targets):
    
    train_inputs = train_inputs[['CreditScore', 'Age', 'Tenure','Balance','NumOfProducts', 'EstimatedSalary']].to_numpy()
    train_targets = train_targets.to_numpy()
    test_inputs = test_inputs[['CreditScore', 'Age', 'Tenure','Balance','NumOfProducts', 'EstimatedSalary']].to_numpy()
    test_targets= test_targets.to_numpy()

    train_inputs = standard_scaler(train_inputs)
    test_inputs = standard_scaler(test_inputs)

    #fit the model to the train data
    weights = logistic_regression_fit(train_inputs, train_targets, threshold = 1e-8)
    #Get the prediction for the test data
    y_pred = logistic_regression_predict(test_inputs, weights)
    #Metrics
    print("Metrics:")

    #CF Matrix
    print("Confusion Matrix")
    cf = two_class_cf_matrix(test_targets, y_pred)
    print(cf)

    #F1 Score
    print("F1_Score")
    f1 = two_class_f1_score(test_targets, y_pred)
    print(f1)


    plt.figure(figsize=(15,7))
    prob_vector = logistic_regression_prediction_probs(test_inputs, weights)
    #print(prob_vector)
    print(len(prob_vector))
    ROC = roc(prob_vector,test_targets, partitions=100)
    #print(ROC)
    plt.scatter(ROC[:,0],ROC[:,1],color='#0F9D58',s=30)
    plt.title('ROC Curve',fontsize=20)
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.ylabel('True Positive Rate',fontsize=16)
    fpr, tpr = ROC[:, 0], ROC[:, 1]
    plt.show()

    #AUC-ROC
    auc = roc_auc(prob_vector,test_targets, partitions=100)
    print('Area under curve={}'.format(auc))
    plt.show()


    plt.show()

