import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from fomlads.evaluate.eval_classification import score, two_class_cf_matrix, roc , roc_auc, two_class_f1_score
from sklearn.ensemble import RandomForestClassifier
import sys
import pickle

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




#Fixing all random state
np.random.seed(123)

input_data = pd.read_csv(sys.argv[1])
input_data.head()

print("Checking Data")
print('\nNull Values \n{}'.format(input_data.isnull().sum()))

print('\nDuplicated values {}'.format(input_data.duplicated().sum()))

print(input_data.head())


#Split Data
from fomlads.model.classification import split_train_test
train_set, test_set = split_train_test(input_data, test_ratio=float(sys.argv[3]))
X_test = test_set.drop([sys.argv[2]], axis=1)
y_test = test_set[sys.argv[2]]

count_class_0, count_class_1 = train_set.Exited.value_counts()

df_class_0 = train_set[train_set[sys.argv[2]] == 0]
df_class_1 = train_set[train_set[sys.argv[2]] == 1]

print("classs count in train")
print("0: ",count_class_0)
print("1: ",count_class_1)


print("Building Classifier")
if sys.argv[4] == "balance":
    print("building classifier with balanced weight")
    X_train = train_set.drop([sys.argv[2]], axis=1)
    y_train = train_set[sys.argv[2]]
    rf_Classifier = RandomForestClassifier(class_weight={0: 1, 1:4}, random_state=123)
    rf_Classifier.fit(X_train, y_train)
    
else:
    print("Not using balanced weight")
    if sys.argv[4] == "undersample":
        df_class_0_under = df_class_0.sample(count_class_1)
        df_train_under = pd.concat([df_class_0_under, df_class_1], axis=0)
        print('Random under-sampling:')
        print(df_train_under.Exited.value_counts())
        X_train = df_train_under.drop([sys.argv[2]], axis=1)
        y_train = df_train_under[sys.argv[2]]
    elif sys.argv[4] == "oversample":
        df_class_1_over = df_class_1.sample(count_class_0, replace=True)
        df_train_over = pd.concat([df_class_0, df_class_1_over], axis=0)
        print('Random over-sampling:')
        print(df_train_over.Exited.value_counts())
        X_train = df_train_over.drop([sys.argv[2]], axis=1)
        y_train = df_train_over[sys.argv[2]]
    else:
        print("no stretegy")
        X_train = train_set.drop([sys.argv[2]], axis=1)
        y_train = train_set[sys.argv[2]]
    rf_Classifier = RandomForestClassifier(random_state=123)
    rf_Classifier.fit(X_train, y_train)

pickle.dump(rf_Classifier, open("RF_model.object", "wb+"))

y_pred = rf_Classifier.predict(X_test)

print("Metrics:")
print("Confusion Matrix")
cf = two_class_cf_matrix(y_test, y_pred)
print(cf)

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

auc = roc_auc(prob_vector,y_test, partitions=100)
print('Area under curve={}'.format(auc))
plt.show()


print("Feature importance")
importance = pd.Series(rf_Classifier.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(importance)
import seaborn as sns
plt.figure(figsize=(10,10))
sns.barplot(x=importance, y=importance.index)
# Add labels to your graph
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title("Freature Importances Plot")
plt.show()