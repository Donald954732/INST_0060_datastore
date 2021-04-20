import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from fomlads.evaluate.eval_classification import two_class_cf_matrix, roc , roc_auc, two_class_f1_score
from sklearn.ensemble import RandomForestClassifier
import sys
import pickle
import seaborn as sns
from preprocessing import pre_process




def rf_main(X_train, y_train, X_test, y_test, weight_balance=True, n_estimators = 100, max_features = "log2"):
    """
    dataset: directory to data set
    target: target column
    train_test_split_ratio: 0<x<1 the ratio for trainigand spliting data
    strategy: strategy to balance the data set imbalanced class
        weight: change the weight of sample in random forest classifier
        undersample: randomly under sample the excess class
        oversample: randomly over sample the class with lower instaces
        None: only use random forest classifier
        This is due to dataset has a imbalanced ratio of classes 
    this will result in traning modals and providing some testing metrics on the model
    the trained model will be saved as RF_model.object for future use
    """
    print("Building Classifier")
    if weight_balance == True:
        print("building classifier with balanced weight")
        rf_Classifier = RandomForestClassifier(class_weight={0: 1, 1:4}, random_state=123, n_estimators=n_estimator, max_features=max_features)
        rf_Classifier.fit(X_train, y_train)
        
    else:
        print("Not using balanced weight")
        rf_Classifier = RandomForestClassifier(random_state=123, n_estimators=n_estimator, max_features=max_features)
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

    prob_vector = rf_Classifier.predict_proba(X_test)[:, 1]

    #AUC-ROC
    auc = roc_auc(prob_vector,y_test, partitions=100)
    print('Area under curve={}'.format(auc))


    return auc
dataframe = pd.read_csv(sys.argv[1])
X_train, y_train, X_test, y_test = pre_process(dataframe, ["RowNumber","CustomerId","Surname"] , "Exited", 0.2, strategy="undersample")
best_auc = 0
best_n_estimator = ""
best_max_feature = "" 
n_estimator_list =[100, 300, 500, 800, 1200, 1500, 1800, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]
max_feature_list = ["auto"]
result = {"auto": []}
for max_feature in max_feature_list:
    for n_estimator in n_estimator_list:
        current_auc_score = rf_main(X_train, y_train, X_test, y_test, weight_balance=False, n_estimators = n_estimator, max_features = max_feature)
        print("Current")
        print(max_feature, n_estimator)
        result[max_feature].append(current_auc_score)
        if current_auc_score > best_auc:
            best_n_estimator = n_estimator
            best_max_feature = max_feature
            best_auc = current_auc_score
print(best_auc, best_n_estimator, best_max_feature)
df = pd.DataFrame(result,index =n_estimator_list)
sns.lineplot(data=df)
plt.show()



