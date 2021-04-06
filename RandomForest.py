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
X_test = test_set.drop(['Exited'], axis=1)
y_test = test_set['Exited']

count_class_0, count_class_1 = train_set.Exited.value_counts()

df_class_0 = train_set[train_set['Exited'] == 0]
df_class_1 = train_set[train_set['Exited'] == 1]

print("classs count in train")
print("0: ",count_class_0)
print("1: ",count_class_1)


print("Building Classifier")
if sys.argv[5] == "balance":
    print("building classifier with balanced weight")
    X_train = train_set.drop(['Exited'], axis=1)
    y_train = train_set['Exited']
    rf_Classifier = RandomForestClassifier(class_weight={0: 1, 1:4}, random_state=123)
    rf_Classifier.fit(X_train, y_train)
    
else:
    print("Not using balanced weight")
    if sys.argv[5] == "undersample":
        df_class_0_under = df_class_0.sample(count_class_1)
        df_train_under = pd.concat([df_class_0_under, df_class_1], axis=0)
        print('Random under-sampling:')
        print(df_train_under.Exited.value_counts())
        X_train = df_train_under.drop(['Exited'], axis=1)
        y_train = df_train_under['Exited']
    elif sys.argv[5] == "oversample":
        df_class_1_over = df_class_1.sample(count_class_0, replace=True)
        df_train_over = pd.concat([df_class_0, df_class_1_over], axis=0)
        print('Random over-sampling:')
        print(df_train_over.Exited.value_counts())
        X_train = df_train_over.drop(['Exited'], axis=1)
        y_train = df_train_over['Exited']
    else:
        print("no stretegy")
        X_train = train_set.drop(['Exited'], axis=1)
        y_train = train_set['Exited']
    rf_Classifier = RandomForestClassifier(random_state=123)
    rf_Classifier.fit(X_train, y_train)

pickle.dump(rf_Classifier, open("RF_model.object", "wb+"))

y_pred = rf_Classifier.predict(X_test)

print("Metrics:")
print("Confusion Matrix")
conf_matrix = cf_matrix(y_test, y_pred)
print(conf_matrix)
print("F1_Score")
print(f1_score(y_test, y_pred))


def true_false_positive(threshold_vector, y_test):
    true_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 1)
    true_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 0)
    false_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 0)
    false_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 1)

    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())

    return tpr, fpr

partitions = 100

def roc(probabilities, y_test, partitions=partitions):
    roc = np.array([])
    for i in range(partitions + 1):
        #print(i)
        
        threshold_vector = np.greater_equal(probabilities, i / partitions)
        tpr, fpr = true_false_positive(threshold_vector, y_test)
        roc = np.append(roc, [fpr, tpr])
        
    return roc.reshape(-1, 2)


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
integrated_roc = 0
for k in range(partitions):
        integrated_roc = integrated_roc + (fpr[k]- fpr[k + 1]) * tpr[k]
print("Area Under Curve of ROC: ", integrated_roc)
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