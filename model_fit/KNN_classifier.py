import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from fomlads.evaluate.eval_classification import two_class_cf_matrix, roc , roc_auc, two_class_f1_score
from sklearn.neighbors import KNeighborsClassifier 
import sys
import pickle
from fomlads.data.external import standard_scaler


def knn_main(X_train, y_train, X_test, y_test):
    """
    Takes inputs and targets split in test and train
    Fit Knn Model 
    Return performance metrics F1, confusion matrix, AUC ROC
    """
    #standardise the inputs
    X_train = standard_scaler(X_train)
    X_test = standard_scaler(X_test)

    print("Building Classifier")
    
   
    print("building classifier with balanced weight")
    knn = KNeighborsClassifier(n_neighbors=90)
    knn.fit(X_train, y_train)
        

    pickle.dump(knn, open("knn_model.object", "wb+"))

    y_pred = knn.predict(X_test)

    #Metrics
    print("Metrics:")

    #CF Matrix
    print("Confusion Matrix")
    cf = two_class_cf_matrix(y_test, y_pred)
    print(cf)

    #F1 Score
    print("F1_Score")
    f1 = two_class_f1_score(y_test, y_pred)
    print(f1)

    prob_vector = knn.predict_proba(X_test)

    plt.figure(figsize=(15,7))
    prob_vector = knn.predict_proba(X_test)[:, 1]
    #print(prob_vector)
    print(len(prob_vector))
    ROC = roc(prob_vector,y_test, partitions=100)
    #print(ROC)
    plt.scatter(ROC[:,0],ROC[:,1],color='#0F9D58',s=30)
    plt.title('ROC Curve',fontsize=20)
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.ylabel('True Positive Rate',fontsize=16)
    fpr, tpr = ROC[:, 0], ROC[:, 1]
    plt.show()

    #AUC-ROC
    auc = roc_auc(prob_vector,y_test, partitions=100)
    print('Area under curve={}'.format(auc))
    plt.show()