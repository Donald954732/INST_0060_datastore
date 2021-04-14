import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from fomlads.evaluate.eval_classification import score, two_class_cf_matrix, roc , roc_auc, two_class_f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
import sys
import pickle
from preprocessing import pre_process
import seaborn as sns


#import preprocess 

##run using python RandomForesy.py dataset target train_test_split_ratio useBalanced_weight
#  dataset: directory to data set
#  target: target column
#  train_test_split_ratio: 0<x<1 the ratio for trainigand spliting data
#  stretegy: stretegy to balance the data set imbalanced class
#       weight: change the weight of sample in random frest classifier
#       undersample: randomly under sample the excess class
#       oversample: randomly over sample the class with lower instaces
#       None: only use random forest classifier
#       This is due to dataset has a imbalanced ratio of classes 
#  this will result in traning modals and providing some testing metrics on the model
#  the trained model will be saved as RF_model.object for future use
#
## example
# python RandomForest.py processed.csv Exited 0.2 balance 




def rf_main(X_train, y_train, X_test, y_test, weight_balance=True):
    print("Building Classifier")
    if weight_balance == True:
        print("building classifier with balanced weight")
        rf_Classifier = KNeighborsClassifier(n_neighbors=90)
        rf_Classifier.fit(X_train, y_train)
        
    else:
        print("Not using balanced weight")
        rf_Classifier = KNeighborsClassifier(n_neighbors=90)
        rf_Classifier.fit(X_train, y_train)

    pickle.dump(rf_Classifier, open("RF_model.object", "wb+"))

    y_pred = rf_Classifier.predict(X_test)

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

    prob_vector = rf_Classifier.predict_proba(X_test)

    plt.figure(figsize=(15,7))
    prob_vector = rf_Classifier.predict_proba(X_test)[:, 1]
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

    ##Feature Importance
    print("Feature importance")
    importance = pd.Series(rf_Classifier.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(importance)
    
    plt.figure(figsize=(10,10))
    sns.barplot(x=importance, y=importance.index)
    # Add labels to your graph
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.title("Freature Importances Plot")
    plt.show()

if __name__ == '__main__':
    np.random.seed(123)
    '''
    column_to_drop = list(input("Input the column to drop :").split(','))
    word_label = list(input("Input the word label columns ").split(','))
    target_column = input("Please Input Target Column: ")
    test_ratio = float(input("Please Input train test ratio: "))
    stretegy = input("Please Input Class Balancing Stretegy:")
    weight_balance_input = input("Balace Weight")
    weight_balance = True if weight_balance_input == "True" else False
    '''
    ##For Testing
    column_to_drop = ['RowNumber', 'Surname', 'CustomerId']
    word_label = ['Gender', 'Geography']
    target_column = "Exited"
    test_ratio = 0.2
    stretegy = "oversample"
    weight_balance = False
    ##
    input_data = pd.read_csv(sys.argv[1])
    input_data.head()
    X_train, y_train, X_test, y_test = pre_process(input_data, column_to_drop, word_label, target_column, test_ratio, stretegy)
    
    rf_main(X_train, y_train, X_test, y_test, weight_balance)